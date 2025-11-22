"""
Sequential Adjustment HITL Agent - Handles multiple user corrections during a single session.

Usage:
    python SequentialAdjustmentAgent.py

Features:
- User can input initial task
- Agent processes with periodic interrupt checks
- User can send multiple sequential adjustments
- All adjustments are combined and maintained in context
- Final output includes all adjustments
- Session history reflects all user inputs
- Agent maintains conversation context throughout

Follows official LangGraph HITL interrupt pattern:
https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, TypedDict, Annotated, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

load_dotenv()


class SequentialAgentState(TypedDict):
    """State schema following LangGraph documentation pattern."""
    messages: Annotated[list[BaseMessage], add_messages]
    initial_goal: str
    current_goal: str
    all_adjustments: List[str]  # Track all sequential adjustments
    adjustment_count: int
    output: str
    task_started_at: str
    last_adjustment_at: Optional[str]
    continue_checking: bool  # Flag to continue checking for more adjustments


class SequentialAdjustmentAgent:
    """
    Agent that handles multiple sequential adjustments during a single session.
    
    Follows LangGraph interrupt() pattern for HITL interactions.
    Maintains conversation context and combines all adjustments into final output.
    """
    
    def _is_adjustment(self, new_input: str, current_goal: str) -> bool:
        """
        Determine if new input is an adjustment to current task or a new task.
        
        Returns True if it's an adjustment (should be combined), False if it's a new task.
        """
        new_input_lower = new_input.strip().lower()
        
        # Adjustment indicators - these suggest it's an addition/modification to current task
        adjustment_indicators = [
            "also", "and", "add", "include", "make it", "change it to", 
            "change to", "update", "modify", "adjust", "edit", "expand",
            "elaborate", "explain more", "more", "further", "additionally",
            "use", "specifically", "with", "implement", "ensure"
        ]
        
        # Check if input starts with adjustment indicators
        for indicator in adjustment_indicators:
            if new_input_lower.startswith(indicator):
                return True
        
        # If input is short and doesn't look like a complete new task, treat as adjustment
        if len(new_input.strip().split()) <= 8:
            # Check if it doesn't contain a complete sentence/question structure
            if not any(char in new_input for char in ['?', '!']) and len(new_input.strip()) < 60:
                return True
        
        # If input is a complete standalone task (longer, has structure), treat as new task
        if len(new_input.strip()) > 60:
            # Check if it's clearly a new topic (starts with capital, has question mark)
            if new_input.strip()[0].isupper() and '?' in new_input:
                return False
        
        # Default: treat as adjustment if it's relatively short
        return len(new_input.strip().split()) <= 10
    
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
        """Build graph with interrupt capability for sequential adjustments."""
        workflow = StateGraph(SequentialAgentState)
        
        workflow.add_node("check_input", self._check_for_adjustment)
        workflow.add_node("process", self._process_task)
        workflow.add_node("generate", self._generate_output)
        
        workflow.set_entry_point("check_input")
        workflow.add_conditional_edges(
            "check_input",
            self._route_after_check,
            {
                "process": "process",
                "adjust": "check_input",  # Loop back if adjustment received
            }
        )
        workflow.add_conditional_edges(
            "process",
            self._route_after_process,
            {
                "adjust": "check_input",  # Loop back if adjustment received
                "generate": "generate",   # Proceed to generate if no adjustment
            }
        )
        workflow.add_edge("generate", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _route_after_check(self, state: SequentialAgentState) -> str:
        """Route based on whether adjustment was received."""
        # Check if we should continue checking for more adjustments
        if state.get("continue_checking", False):
            # Clear the flag and loop back to check for more adjustments
            state["continue_checking"] = False
            return "adjust"  # Loop back to check_input
        
        # No more adjustments expected, proceed to process
        return "process"
    
    def _route_after_process(self, state: SequentialAgentState) -> str:
        """Route after process - check if adjustment was received during work."""
        # If continue_checking is True, it means an adjustment was received during process
        if state.get("continue_checking", False):
            # Clear flag and loop back to check for more adjustments
            state["continue_checking"] = False
            print("[DEBUG] Adjustment received during process, routing back to check_input")
            return "adjust"  # Loop back to check_input
        
        # No adjustment received, proceed to generate
        print("[DEBUG] Process completed, routing to generate")
        return "generate"
    
    def _check_for_adjustment(self, state: SequentialAgentState) -> SequentialAgentState:
        """
        Check for new adjustments - uses interrupt() to pause and check.
        Follows LangGraph documentation pattern.
        """
        current_goal = state.get("current_goal", "")
        initial_goal = state.get("initial_goal", "")
        messages = state.get("messages", [])
        all_adjustments = state.get("all_adjustments", [])
        
        # If this is first run, set initial goal
        if not initial_goal and messages:
            latest_msg = messages[-1]
            if isinstance(latest_msg, HumanMessage):
                goal = latest_msg.content
                state["initial_goal"] = goal
                state["current_goal"] = goal
                state["all_adjustments"] = []
                state["adjustment_count"] = 0
                state["task_started_at"] = datetime.now().isoformat()
                state["continue_checking"] = True  # Allow checking for adjustments
                return state
        
        # Check for new input using interrupt - this pauses execution
        # Following LangGraph documentation: interrupt() pauses, Command(resume=value) resumes
        if current_goal:
            prompt = f"Current task: '{current_goal[:70]}...' | Send adjustment to modify, or press Enter to continue."
        else:
            prompt = "Send adjustment or press Enter to continue."
        
        new_input = interrupt(prompt)
        
        # If new input received
        if new_input and new_input.strip():
            # Determine if it's an adjustment or new task
            is_adjustment = self._is_adjustment(new_input.strip(), current_goal)
            
            if is_adjustment:
                # Sequential adjustment - combine with current goal
                adjustment = new_input.strip()
                all_adjustments.append(adjustment)
                
                # Build combined goal: initial + all adjustments
                combined_parts = [initial_goal] + all_adjustments
                combined_goal = " ".join(combined_parts)
                
                print(f"\n[ADJUSTMENT #{len(all_adjustments)}] Received: '{adjustment}'")
                print(f"[UPDATE] Combined goal: '{combined_goal}'")
                
                state["current_goal"] = combined_goal
                state["all_adjustments"] = all_adjustments
                state["adjustment_count"] = len(all_adjustments)
                state["last_adjustment_at"] = datetime.now().isoformat()
                state["continue_checking"] = True  # Continue checking for more adjustments
                
                # Add message to conversation history
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(HumanMessage(content=adjustment))
            else:
                # New task - reset everything
                print(f"\n[NEW TASK] Received: '{new_input}'")
                print(f"[RESET] Starting new task (previous task cancelled)")
                
                state["initial_goal"] = new_input.strip()
                state["current_goal"] = new_input.strip()
                state["all_adjustments"] = []
                state["adjustment_count"] = 0
                state["task_started_at"] = datetime.now().isoformat()
                state["continue_checking"] = False  # New task, proceed to process
                
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append(HumanMessage(content=new_input.strip()))
        else:
            # No input received (user pressed Enter), clear flag to proceed to process
            state["continue_checking"] = False
            print("[DEBUG] No input received, proceeding to process")
        
        return state
    
    def _process_task(self, state: SequentialAgentState) -> SequentialAgentState:
        """
        Process the current task - simulate work with periodic interrupt checks.
        Uses interrupt() pattern from LangGraph documentation.
        """
        goal = state.get("current_goal", "")
        if not goal:
            return state
        
        print(f"\n[PROCESS] Starting to work on: {goal}")
        print("[INFO] You can send adjustments anytime during the next 60 seconds!")
        
        # Simulate 60 seconds of work, but check for adjustments periodically
        work_duration = 60.0
        check_interval = 10.0  # Check every 10 seconds
        elapsed = 0.0
        
        while elapsed < work_duration:
            remaining = work_duration - elapsed
            print(f"[WORK] Processing... ({elapsed:.0f}s / {work_duration:.0f}s elapsed)")
            
            # Use interrupt() to pause and allow adjustments - follows documentation pattern
            new_input = interrupt(f"Working on task ({elapsed:.0f}s elapsed). Send adjustment or Enter to continue.")
            
            # If new input received
            if new_input and new_input.strip():
                is_adjustment = self._is_adjustment(new_input.strip(), goal)
                
                messages = state.get("messages", [])
                messages.append(HumanMessage(content=new_input.strip()))
                
                if is_adjustment:
                    # Sequential adjustment during work
                    all_adjustments = state.get("all_adjustments", [])
                    adjustment = new_input.strip()
                    all_adjustments.append(adjustment)
                    
                    initial_goal = state.get("initial_goal", goal)
                    combined_parts = [initial_goal] + all_adjustments
                    combined_goal = " ".join(combined_parts)
                    
                    print(f"\n[ADJUSTMENT #{len(all_adjustments)}] Received during work: '{adjustment}'")
                    print(f"[UPDATE] Updated goal: '{combined_goal}'")
                    
                    return {
                        **state,
                        "current_goal": combined_goal,
                        "all_adjustments": all_adjustments,
                        "adjustment_count": len(all_adjustments),
                        "last_adjustment_at": datetime.now().isoformat(),
                        "continue_checking": True,  # Continue checking for more adjustments
                        "messages": messages,
                    }
                else:
                    # New task - reset
                    print(f"\n[NEW TASK] Received during work: '{new_input}'")
                    print(f"[RESET] Starting new task")
                    
                    return {
                        **state,
                        "initial_goal": new_input.strip(),
                        "current_goal": new_input.strip(),
                        "all_adjustments": [],
                        "adjustment_count": 0,
                        "task_started_at": datetime.now().isoformat(),
                        "messages": messages,
                    }
            
            # Simulate work
            import time
            sleep_time = min(check_interval, remaining)
            if sleep_time > 0:
                time.sleep(sleep_time)
            elapsed += check_interval
        
        print(f"\n[COMPLETE] Finished processing: {goal}")
        print("[DEBUG] Process node returning, will route to generate")
        # Ensure continue_checking is False so routing goes to generate
        state["continue_checking"] = False
        return state
    
    def _generate_output(self, state: SequentialAgentState) -> SequentialAgentState:
        """
        Generate final output based on combined goal with all adjustments.
        Maintains conversation context from all messages.
        """
        goal = state.get("current_goal", "")
        initial_goal = state.get("initial_goal", "")
        all_adjustments = state.get("all_adjustments", [])
        messages = state.get("messages", [])
        
        if not goal:
            print("[DEBUG] No goal in state, skipping generation")
            return state
        
        print(f"\n[GENERATE] Generating output with all adjustments...")
        print(f"  Initial goal: {initial_goal}")
        if all_adjustments:
            print(f"  Adjustments ({len(all_adjustments)}): {', '.join(all_adjustments)}")
        
        # Create comprehensive prompt that includes all context
        if all_adjustments:
            adjustments_text = "\n".join(f"- {adj}" for adj in all_adjustments)
            prompt = f"""User request: "{initial_goal}"

Additional requirements/adjustments:
{adjustments_text}

Combined task: "{goal}"

Generate the requested content according to ALL the requirements above.
Include everything from the initial request and all subsequent adjustments.
Maintain context and coherence across all requirements.
"""
        else:
            prompt = f"""User request: "{goal}"

Generate the requested content according to the instructions.
"""
        
        try:
            # Use conversation history for context
            context_messages = messages + [HumanMessage(content=prompt)]
            print(f"[DEBUG] Calling LLM with {len(context_messages)} messages")
            response = self.llm.invoke(context_messages)
            
            state["output"] = response.content
            print(f"[DEBUG] Generated output length: {len(response.content)} characters")
        except Exception as e:
            print(f"[ERROR] Failed to generate output: {e}")
            import traceback
            traceback.print_exc()
            state["output"] = f"Error generating output: {str(e)}"
        
        return state
    
    async def run(self, thread_id: str = "sequential_session"):
        """Run sequential adjustment HITL session."""
        config = {"configurable": {"thread_id": thread_id}}
        
        print("=" * 80)
        print("Sequential Adjustment HITL Agent")
        print("=" * 80)
        print("\nInstructions:")
        print("- Enter your initial task/query")
        print("- Agent will process for 60 seconds")
        print("- During processing, you can send MULTIPLE sequential adjustments")
        print("- All adjustments will be combined and included in final output")
        print("- Session history maintains all inputs for context")
        print("- Type 'exit' or 'quit' to stop")
        print("=" * 80)
        print()
        
        # Get initial input
        initial_input = input("Enter your initial task/query: ").strip()
        
        if not initial_input or initial_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            return
        
        # Initialize state
        initial_state: SequentialAgentState = {
            "messages": [HumanMessage(content=initial_input)],
            "initial_goal": initial_input,
            "current_goal": initial_input,
            "all_adjustments": [],
            "adjustment_count": 0,
            "output": "",
            "task_started_at": datetime.now().isoformat(),
            "last_adjustment_at": None,
            "continue_checking": True,  # Start by checking for adjustments
        }
        
        self.graph.update_state(config, initial_state)
        
        # Start processing
        print(f"\n[START] Initial task: {initial_input}")
        
        # Run graph - it will pause at interrupt() calls
        async for event in self.graph.astream(None, config):
            for node, node_state in event.items():
                print(f"[DEBUG] Node executed: {node}")
        
        # Handle interrupts - keep resuming until done
        # Follows LangGraph documentation pattern: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/#agent
        current_state = self.graph.get_state(config)
        done = False
        
        while not done and current_state and current_state.next:
            # Graph is paused at interrupt() - get user input
            user_input = input("\n> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            # Resume with user input - follows LangGraph pattern from documentation
            async for event in self.graph.astream(Command(resume=user_input), config):
                for node, node_state in event.items():
                    print(f"[DEBUG] Node executed after resume: {node}")
            
            # Check state after resume
            current_state = self.graph.get_state(config)
            
            # Continue execution until graph pauses again or completes
            # When a node completes (no next), LangGraph should automatically call routing function
            # for conditional edges, then execute the next node
            while not done and not current_state.next:
                # Node completed - continue execution to trigger routing and next node
                print("[DEBUG] Node completed, continuing execution to trigger routing...")
                async for event in self.graph.astream(None, config):
                    for node, node_state in event.items():
                        print(f"[DEBUG] Continuing execution - Node: {node}")
                
                # Check state again
                current_state = self.graph.get_state(config)
                
                # If graph is paused (has next), break inner loop to handle interrupt
                if current_state.next:
                    break
                
                # Check if we're done
                final_check = self.graph.get_state(config)
                if final_check.values.get("output"):
                    # Output generated, we're done
                    done = True
                    break
        
        # Get final results
        final_state = self.graph.get_state(config)
        result = final_state.values
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Initial Goal: {result.get('initial_goal', 'N/A')}")
        print(f"Total Adjustments: {result.get('adjustment_count', 0)}")
        if result.get('all_adjustments'):
            print("Adjustments:")
            for i, adj in enumerate(result.get('all_adjustments', []), 1):
                print(f"  {i}. {adj}")
        print(f"Final Combined Goal: {result.get('current_goal', 'N/A')}")
        print(f"Output Length: {len(result.get('output', ''))} characters")
        print("\nSession History:")
        print("-" * 80)
        messages = result.get('messages', [])
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                print(f"{i}. User: {msg.content}")
        print("-" * 80)
        print("\nFinal Output:")
        print("-" * 80)
        print(result.get('output', 'No output generated'))
        print("-" * 80)


async def main():
    """Main entry point."""
    try:
        agent = SequentialAdjustmentAgent()
        await agent.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

