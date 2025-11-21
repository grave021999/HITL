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
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
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
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph with interrupt capability using interrupt_after."""
        workflow = StateGraph(AgentState)

        workflow.add_node("check_new_input", self._check_new_input)
        workflow.add_node("plan", self._plan_task)
        workflow.add_node("wait_for_input", self._wait_for_input)
        workflow.add_node("generate", self._generate_content)

        workflow.set_entry_point("check_new_input")

        workflow.add_conditional_edges(
            "check_new_input",
            self._route_after_check,
            {
                "new_input": "plan",
                "continue": "plan",
                "cancelled": "plan",
            },
        )

        workflow.add_edge("plan", "wait_for_input")
        
        # After wait_for_input, check if we need to recheck for new input
        workflow.add_conditional_edges(
            "wait_for_input",
            self._route_after_wait,
            {
                "generate": "generate",
                "recheck": "check_new_input",  # Recheck if new input detected
            },
        )
        
        workflow.add_edge("generate", END)

        # Compile with interrupt_after to pause after wait_for_input node
        # This allows human input to be injected before generation
        # The graph will pause AFTER wait_for_input executes, allowing us to add new messages
        app = workflow.compile(
            checkpointer=self.memory,
            interrupt_after=["wait_for_input"]
        )
        return app

    def _check_new_input(self, state: AgentState) -> AgentState:
        """Detect new human instructions and cancel previous goal if needed."""
        messages = state.get("messages", [])
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]

        if not human_msgs:
            return state

        latest = human_msgs[-1].content
        current_goal = state.get("current_goal", "")

        # First instruction
        if current_goal == "" and len(human_msgs) == 1:
            return {
                **state,
                "current_goal": latest,
                "task_started_at": datetime.now().isoformat(),
                "cancelled": False,
            }

        # New instruction different from current goal => interrupt
        if latest != current_goal:
            print(f"[CANCEL] Interrupt detected. Old goal: '{current_goal}' -> New: '{latest}'")
            return {
                **state,
                "previous_goal": current_goal,
                "current_goal": latest,
                "cancelled": True,
                "task_cancelled_at": datetime.now().isoformat(),
                "output": "",
            }

        return state

    def _route_after_check(self, state: AgentState) -> Literal["new_input", "continue", "cancelled"]:
        """Route based on interrupt check."""
        human_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]

        if len(human_msgs) >= 2 and human_msgs[-1] != human_msgs[-2]:
            return "new_input"

        if state.get("cancelled"):
            return "cancelled"

        return "continue"

    def _plan_task(self, state: AgentState) -> AgentState:
        """Plan the task based on current goal."""
        # Don't clear cancelled flag - we want to preserve it to show previous task was cancelled
        # The cancelled flag indicates the PREVIOUS task was cancelled, not the current one
        print(f"[PLAN] Planning task: {state['current_goal']}")
        return state

    def _route_after_wait(self, state: AgentState) -> Literal["generate", "recheck"]:
        """Route after wait - check if new input was added during interrupt."""
        human_msgs = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
        current_goal = state.get("current_goal", "")

        # If we have multiple messages and the latest is different, recheck
        if len(human_msgs) > 1:
            latest_goal = human_msgs[-1]  # Already a string from list comprehension
            if latest_goal != current_goal:
                print(f"[ROUTE] New input detected, routing to recheck...")
                return "recheck"

        return "generate"

    def _wait_for_input(self, state: AgentState) -> AgentState:
        """
        Wait node - execution pauses AFTER this node due to interrupt_after.
        This allows us to inject new messages before generation.
        """
        current_goal = state.get("current_goal", "")
        print(f"[WAIT] Node executed. Current goal: '{current_goal}'")
        print("[INTERRUPT] Graph will pause here (interrupt_after) to allow input...")
        return state

    def _generate_content(self, state: AgentState) -> AgentState:
        """Generate content based on current goal."""
        goal = state.get("current_goal", "")
        if not goal:
            return state

        prompt = f"""
User request: "{goal}"

Generate the requested content EXACTLY according to instructions.
If the user changed their request, do ONLY what the latest request says.
"""
        print(f"[GENERATE] Generating: {goal}")
        result = self.llm.invoke(prompt)

        return {**state, "output": result.content}

    async def process_with_interrupt(
        self,
        thread_id: str,
        initial_message: str,
        interrupt_after_seconds: float = None,
        interrupt_message: str = None,
    ) -> dict:
        """
        Process a task with optional interrupt using LangGraph's interrupt mechanism.
        
        This method properly uses LangGraph's interrupt() function and Command objects
        to handle human-in-the-loop scenarios.
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

        if not (interrupt_after_seconds and interrupt_message):
            # No interrupt - run normally
            print(f"[START] Task started: {initial_message}")
            async for event in self.graph.astream(None, config):
                for node, node_state in event.items():
                    print(f"[NODE] {node} executed")
            
            final_state = self.graph.get_state(config)
            return final_state.values

        # With interrupt: use LangGraph's interrupt mechanism
        print(f"[START] Task started: {initial_message}")
        
        # Start the graph execution - it will pause after wait_for_input (due to interrupt_after)
        async for event in self.graph.astream(None, config):
            for node, node_state in event.items():
                print(f"[NODE] {node} executed")
        
        # Check if graph is in interrupted state
        current_state = self.graph.get_state(config)
        if current_state.next:
            # Graph is paused/interrupted
            print(f"[WAIT] Simulating work in progress... ({interrupt_after_seconds}s)")
            await asyncio.sleep(interrupt_after_seconds)
            
            # Add interrupt message to state before resuming
            print(f"[INTERRUPT] Sending interrupt: {interrupt_message}")
            current_messages = current_state.values.get("messages", [])
            current_messages.append(HumanMessage(content=interrupt_message))
            
            # Update state with new message
            self.graph.update_state(
                config,
                {"messages": current_messages}
            )
            
            # Resume execution with Command - graph will continue and check for new input
            print("[RESUME] Resuming execution with new input...")
            max_resumes = 3  # Prevent infinite loops
            resume_count = 0
            
            while resume_count < max_resumes:
                async for event in self.graph.astream(Command(resume=True), config):
                    for node, node_state in event.items():
                        print(f"[NODE] {node} executed (after resume #{resume_count + 1})")
                
                # Check if graph is still paused (might pause again after replan)
                current_state = self.graph.get_state(config)
                if current_state.next:
                    # Graph paused again (after replan), resume to continue to generation
                    resume_count += 1
                    print(f"[RESUME] Resuming again (resume #{resume_count + 1}) to continue generation...")
                else:
                    # Graph completed
                    break

        # Get final state
        final_state = self.graph.get_state(config)
        return final_state.values
