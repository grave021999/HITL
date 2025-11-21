"""
LangGraph agent with Human-in-the-Loop interrupt capability using official LangGraph interrupt pattern.

Uses LangGraph's interrupt() function and interrupt_before/interrupt_after compilation options
to properly pause execution and wait for human input.
"""

import os
import asyncio
from typing import TypedDict, Annotated, Literal
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_goal: str
    previous_goal: str
    cancelled: bool
    output: str
    task_started_at: str
    task_cancelled_at: str


# Define AskHuman tool for HITL questions
class AskHuman(BaseModel):
    """Ask the human a question and wait for their response."""
    question: str


class ReportAgent:
    """Agent that can generate reports and be interrupted by human input using LangGraph interrupts."""

    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided")

        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        self.llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
        )
        
        # Bind tools to model - including AskHuman for HITL
        # The model can call AskHuman when it needs to ask the user a question
        self.llm = self.llm.bind_tools([AskHuman])
        
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with interrupt capability using interrupt() function."""
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", self._call_agent)
        workflow.add_node("ask_human", self._ask_human)  # HITL node that calls interrupt()
        workflow.add_node("generate", self._generate_content)

        workflow.set_entry_point("agent")

        # Route based on whether agent called a tool (AskHuman) or finished
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "ask_human": "ask_human",
                "generate": "generate",
                "end": END,
            },
        )

        # After asking human, go back to agent with the response
        workflow.add_edge("ask_human", "agent")
        
        workflow.add_edge("generate", END)

        # Compile with checkpointer - interrupts are handled via interrupt() calls in nodes
        app = workflow.compile(
            checkpointer=self.memory
        )
        return app


    def _call_agent(self, state: AgentState) -> AgentState:
        """Call the LLM agent - it may decide to ask a question via AskHuman tool."""
        messages = state.get("messages", [])
        response = self.llm.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: AgentState) -> Literal["ask_human", "generate", "end"]:
        """Determine next step based on agent's response."""
        messages = state.get("messages", [])
        last_message = messages[-1]
        
        # If agent called AskHuman tool, route to ask_human node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if last_message.tool_calls[0]["name"] == "AskHuman":
                return "ask_human"
        
        # If no tool calls and we have a goal, generate content
        if state.get("current_goal"):
            return "generate"
        
        # Otherwise end
        return "end"
    
    def _ask_human(self, state: AgentState) -> AgentState:
        """
        Ask human a question using interrupt() - follows documentation pattern.
        When agent calls AskHuman tool, this node is triggered.
        It calls interrupt() with the question, waits for user input,
        and returns a tool message with the response.
        """
        messages = state.get("messages", [])
        last_message = messages[-1]
        
        # Extract the question from the tool call
        tool_call_id = last_message.tool_calls[0]["id"]
        ask = AskHuman.model_validate(last_message.tool_calls[0]["args"])
        question = ask.question
        
        # Call interrupt() with the question - this pauses execution
        # When resumed with Command(resume=value), that value is returned here
        print(f"[ASK_HUMAN] Question: {question}")
        answer = interrupt(question)
        
        # Create tool message with the answer
        tool_message = ToolMessage(
            tool_call_id=tool_call_id,
            content=answer if answer else ""
        )
        
        return {"messages": [tool_message]}

    def _generate_content(self, state: AgentState) -> AgentState:
        """Generate final content based on conversation."""
        messages = state.get("messages", [])
        
        # Get the original goal from state
        goal = state.get("current_goal", "")
        
        # Call agent one more time to generate final output
        # The agent has all context including any HITL responses
        response = self.llm.invoke(messages)
        
        return {
            **state,
            "output": response.content,
            "messages": [response]
        }

    async def process_with_interrupt(
        self,
        thread_id: str,
        initial_message: str,
        interrupt_after_seconds: float = None,
        interrupt_message: str = None,
    ) -> dict:
        """
        Process a task with HITL interrupt capability.
        
        The agent may ask questions via AskHuman tool, which triggers interrupt().
        When interrupted, waits for user input and resumes with that input.
        """
        config = {"configurable": {"thread_id": thread_id}}

        # Initialize with first message
        initial_state = {
            "messages": [HumanMessage(content=initial_message)],
            "current_goal": initial_message,
            "previous_goal": "",
            "cancelled": False,
            "output": "",
            "task_started_at": datetime.now().isoformat(),
            "task_cancelled_at": "",
        }

        # Update state
        self.graph.update_state(config, initial_state)

        print(f"[START] Task started: {initial_message}")
        
        # Run graph - it will pause when agent calls AskHuman tool and interrupt() is triggered
        async for event in self.graph.astream(None, config):
            for node, node_state in event.items():
                print(f"[NODE] {node} executed")
        
        # Check if graph is interrupted (waiting for user input at ask_human node)
        current_state = self.graph.get_state(config)
        
        # Keep resuming until graph completes (agent may ask multiple questions)
        while current_state.next:
            # Graph is paused at interrupt() call in ask_human node
            if interrupt_after_seconds and interrupt_message:
                # Test mode: simulate delay then use hardcoded message
                print(f"[WAIT] Simulating work in progress... ({interrupt_after_seconds}s)")
                await asyncio.sleep(interrupt_after_seconds)
                user_input = interrupt_message
                print(f"[INTERRUPT] Using test input: {user_input}")
                interrupt_after_seconds = None  # Only simulate delay once
            else:
                # Real HITL mode: wait for actual user input
                print("[INTERRUPT] Agent is asking a question - waiting for your response...")
                user_input = input("Your answer: ").strip()
            
            # Resume with user input - this becomes the return value of interrupt()
            print(f"[RESUME] Resuming with: {user_input}")
            async for event in self.graph.astream(Command(resume=user_input), config):
                for node, node_state in event.items():
                    print(f"[NODE] {node} executed (after resume)")
            
            # Check if graph paused again (agent may ask another question)
            current_state = self.graph.get_state(config)

        # Get final state
        final_state = self.graph.get_state(config)
        return final_state.values
