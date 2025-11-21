"""
Interactive HITL Agent - Allows unsolicited user input to interrupt and cancel ongoing tasks.

Usage:
    python HITL_Agent.py

Features:
- User can input ANY query
- Agent processes for 60 seconds
- During processing, user can send NEW query to interrupt
- New query cancels previous task and switches immediately
- Works for any topic, not hardcoded
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, TypedDict, Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    """State schema following LangGraph documentation pattern."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_goal: str
    previous_goal: str
    cancelled: bool
    output: str
    task_started_at: str
    task_cancelled_at: Optional[str]


class InteractiveHITLAgent:
    """Interactive agent that can be interrupted by unsolicited user input."""
    
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
    
    def _build_graph(self):
        """Build graph with interrupt capability for unsolicited input."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("check_input", self._check_for_new_input)
        workflow.add_node("process", self._process_task)
        workflow.add_node("generate", self._generate_output)
        
        workflow.set_entry_point("check_input")
        workflow.add_conditional_edges(
            "check_input",
            self._route_after_check,
            {
                "process": "process",
                "new_task": "check_input",  # Loop back if new task received
            }
        )
        workflow.add_edge("process", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _route_after_check(self, state: AgentState) -> str:
        """Route based on whether new input was received."""
        if state.get("cancelled", False):
            return "new_task"  # New task received, re-check
        return "process"  # Continue with current task
    
    def _check_for_new_input(self, state: AgentState) -> AgentState:
        """Check for new unsolicited input - uses interrupt() to pause and check."""
        current_goal = state.get("current_goal", "")
        messages = state.get("messages", [])
        
        # If this is first run, set initial goal
        if not current_goal and messages:
            latest_msg = messages[-1]
            if isinstance(latest_msg, HumanMessage):
                state["current_goal"] = latest_msg.content
                state["task_started_at"] = datetime.now().isoformat()
                state["cancelled"] = False
                state["previous_goal"] = ""
                return state
        
        # Check for new input using interrupt - this pauses execution
        # During pause, user can send new input which will be returned here
        prompt = f"Current task: '{current_goal[:60]}...' | Send new task to interrupt, or press Enter to continue."
        new_input = interrupt(prompt)
        
        # If new input received and different from current goal, cancel previous task
        if new_input and new_input.strip() and new_input.strip() != current_goal:
            print(f"\n[INTERRUPT] New input received: '{new_input}'")
            print(f"[CANCEL] Cancelling previous task: '{current_goal}'")
            
            state["previous_goal"] = current_goal
            state["current_goal"] = new_input.strip()
            state["cancelled"] = True
            state["task_cancelled_at"] = datetime.now().isoformat()
            state["output"] = ""  # Clear previous output
            
            # Add new message
            if "messages" not in state:
                state["messages"] = []
            state["messages"].append(HumanMessage(content=new_input.strip()))
        else:
            state["cancelled"] = False
        
        return state
    
    def _process_task(self, state: AgentState) -> AgentState:
        """Process the current task - simulate work for 60 seconds with periodic interrupt checks."""
        goal = state.get("current_goal", "")
        if not goal:
            return state
        
        print(f"\n[PROCESS] Starting to work on: {goal}")
        print("[INFO] You can send a new query anytime during the next 60 seconds to interrupt!")
        
        # Simulate 60 seconds of work, but check for interrupts periodically
        # Following documentation pattern: interrupt() pauses, resume with Command(resume=value)
        work_duration = 60.0
        check_interval = 10.0  # Check every 10 seconds
        elapsed = 0.0
        
        while elapsed < work_duration:
            remaining = work_duration - elapsed
            print(f"[WORK] Processing... ({elapsed:.0f}s / {work_duration:.0f}s elapsed)")
            
            # Use interrupt() to pause and allow new input - follows documentation pattern
            # When resumed with Command(resume=value), that value is returned here
            new_input = interrupt(f"Working on '{goal[:40]}...' ({elapsed:.0f}s elapsed). Send new task to interrupt, or Enter to continue.")
            
            # If new input received and different, cancel and return immediately
            if new_input and new_input.strip() and new_input.strip() != goal:
                print(f"\n[INTERRUPT] New task received during work: '{new_input}'")
                print(f"[CANCEL] Stopping current work on: '{goal}'")
                
                # Update state with new task
                messages = state.get("messages", [])
                messages.append(HumanMessage(content=new_input.strip()))
                
                return {
                    **state,
                    "previous_goal": goal,
                    "current_goal": new_input.strip(),
                    "cancelled": True,
                    "task_cancelled_at": datetime.now().isoformat(),
                    "output": "",
                    "messages": messages,
                }
            
            # Simulate work (synchronous sleep - nodes should be sync unless doing async I/O)
            import time
            sleep_time = min(check_interval, remaining)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elapsed += check_interval
        
        print(f"\n[COMPLETE] Finished processing: {goal}")
        return state
    
    def _generate_output(self, state: AgentState) -> AgentState:
        """Generate final output based on current goal."""
        goal = state.get("current_goal", "")
        
        if not goal:
            return state
        
        print(f"\n[GENERATE] Generating output for: {goal}")
        
        # Create prompt
        prompt = f"""User request: "{goal}"

Generate the requested content according to the instructions.
If the user changed their request, focus ONLY on the latest request.
Be concise and focused on what was requested.
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        state["output"] = response.content
        return state
    
    async def run_interactive(self, thread_id: str = "interactive_session"):
        """Run interactive HITL session."""
        config = {"configurable": {"thread_id": thread_id}}
        
        print("=" * 80)
        print("Interactive HITL Agent")
        print("=" * 80)
        print("\nInstructions:")
        print("- Enter any query/task for the agent")
        print("- Agent will process for 60 seconds")
        print("- During processing, you can send a NEW query to interrupt")
        print("- New query will cancel previous task and switch immediately")
        print("- Type 'exit' or 'quit' to stop")
        print("=" * 80)
        print()
        
        # Get initial input
        initial_input = input("Enter your first task/query: ").strip()
        
        if not initial_input or initial_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            return
        
        # Initialize state - following TypedDict pattern from documentation
        initial_state: AgentState = {
            "messages": [HumanMessage(content=initial_input)],
            "current_goal": initial_input,
            "previous_goal": "",
            "cancelled": False,
            "output": "",
            "task_started_at": datetime.now().isoformat(),
            "task_cancelled_at": None,
        }
        
        self.graph.update_state(config, initial_state)
        
        # Start processing
        print(f"\n[START] Task: {initial_input}")
        
        # Run graph - it will pause at interrupt() calls
        async for event in self.graph.astream(None, config):
            pass
        
        # Handle interrupts - keep resuming until done
        current_state = self.graph.get_state(config)
        
        while current_state.next:
            # Graph is paused at interrupt() - get user input
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            # Resume with user input (empty string = continue, non-empty = new task)
            async for event in self.graph.astream(Command(resume=user_input), config):
                pass
            
            current_state = self.graph.get_state(config)
        
        # Get final results
        final_state = self.graph.get_state(config)
        result = final_state.values
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Previous Goal: {result.get('previous_goal', 'N/A')}")
        print(f"Current Goal: {result.get('current_goal', 'N/A')}")
        print(f"Cancelled: {result.get('cancelled', False)}")
        print(f"Output Length: {len(result.get('output', ''))} characters")
        print("\nOutput:")
        print("-" * 80)
        print(result.get('output', 'No output generated'))
        print("-" * 80)


async def main():
    """Main entry point."""
    try:
        agent = InteractiveHITLAgent()
        await agent.run_interactive()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
