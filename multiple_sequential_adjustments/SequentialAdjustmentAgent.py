"""
Sequential Adjustment HITL Agent

Usage:
    python SequentialAdjustmentAgent.py

What this does
--------------
- User gives an initial task (any scenario).
- Agent collects zero or more sequential "adjustments" from the user
  using LangGraph's human-in-the-loop `interrupt` + `Command(resume=...)` pattern.
- Adjustments are merged with the initial goal into a single combined goal.
- The final LLM call is made with full conversation history.
- Final output is generated using ALL requirements (initial + adjustments).

This matches the official LangGraph Human-in-the-loop pattern:
https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/

You can test with any scenario, for example:

1. Initial: "Create a Python REST API for user management"
2. Adjustment: "Add JWT authentication"
3. Adjustment: "Use FastAPI framework specifically"

But nothing is hardcoded for this; it's just one example.
"""

import os
from datetime import datetime, timezone
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

load_dotenv()


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


class SequentialAdjustmentState(TypedDict, total=False):
    """State schema compatible with LangGraph + extra fields for our use-case.

    Keys:
      - messages: full chat history (user + assistant). Required by LangGraph.
      - base_goal: initial task from the user.
      - current_goal: base_goal + all adjustments combined.
      - adjustments: list of adjustment strings (in order).
      - adjustment_count: len(adjustments).
      - output: final generated content from the LLM.
      - created_at / last_updated_at: simple timestamps.
      - done_collecting: True when user is finished giving adjustments.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    base_goal: str
    current_goal: str
    adjustments: List[str]
    adjustment_count: int
    output: str
    created_at: str
    last_updated_at: str
    done_collecting: bool


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------


class SequentialAdjustmentAgent:
    """
    Agent that supports multiple sequential user adjustments in a single session,
    using LangGraph's human-in-the-loop `interrupt()` mechanism.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set in environment or passed in.")

        model_name = model_name or os.getenv(
            "ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"
        )

        self.llm = ChatAnthropic(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
        )

        # LangGraph checkpointer so we can pause/resume (HITL)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Heuristic: is this new input an *adjustment* or a *new task*?
    # ------------------------------------------------------------------

    def _is_adjustment(self, new_input: str, current_goal: str) -> bool:
        """
        Heuristic to decide whether new_input is an adjustment to the current task
        or a completely new task that should replace it.

        - Short, modifier-like inputs ("add", "use", "include", etc.) are
          treated as adjustments.
        - Long inputs that look like fresh standalone tasks, especially with
          low word overlap with the current goal, are treated as new tasks.
        """
        new_input = new_input.strip()
        new_lower = new_input.lower()
        goal_lower = current_goal.lower().strip()

        # If there is no current goal, treat this as a new task.
        if not goal_lower:
            return False

        # Phrases that usually mean "throw away old, start something else"
        cancellation_indicators = [
            "actually",
            "instead",
            "forget",
            "ignore",
            "cancel",
            "stop",
            "scratch that",
            "never mind",
            "change to",
            "switch to",
            "do this instead",
            "no wait",
            "focus on",
            "just ",
        ]
        for ind in cancellation_indicators:
            if new_lower.startswith(ind):
                return False

        # Phrases that usually mean "refine/extend this task"
        adjustment_indicators = [
            "also",
            "and",
            "add",
            "include",
            "update",
            "modify",
            "adjust",
            "edit",
            "expand",
            "elaborate",
            "more",
            "further",
            "additionally",
            "use",
            "specifically",
            "with",
            "implement",
            "ensure",
            "make sure",
            "don't forget",
            "plus",
        ]
        for ind in adjustment_indicators:
            if new_lower.startswith(ind):
                return True

        # Very short inputs are often "quick tweaks"
        if len(new_input.split()) <= 5 and len(new_input) < 60:
            return True

        # If there is almost no word overlap with the current goal and the
        # new input is substantial, treat as a new task.
        goal_words = set(goal_lower.split())
        new_words = set(new_lower.split())
        overlap = len(goal_words.intersection(new_words))

        if overlap < 2 and len(new_input.split()) > 8:
            return False

        # Default bias: short-ish things are adjustments, long are new tasks.
        return len(new_input.split()) <= 10

    # ------------------------------------------------------------------
    # Graph definition
    # ------------------------------------------------------------------

    def _build_graph(self):
        workflow = StateGraph(SequentialAdjustmentState)

        workflow.add_node("init_goal", self._init_goal)
        workflow.add_node("collect_adjustments", self._collect_adjustments)
        workflow.add_node("generate_output", self._generate_output)

        # Start at init_goal
        workflow.add_edge(START, "init_goal")
        workflow.add_edge("init_goal", "collect_adjustments")

        # From collect_adjustments, either:
        #   - loop to collect more adjustments
        #   - or go to generate_output
        workflow.add_conditional_edges(
            "collect_adjustments",
            self._route_after_collect,
            {"more": "collect_adjustments", "generate": "generate_output"},
        )

        workflow.add_edge("generate_output", END)

        return workflow.compile(checkpointer=self.memory)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _init_goal(self, state: SequentialAdjustmentState) -> SequentialAdjustmentState:
        """Initialize base_goal/current_goal from the latest HumanMessage."""
        messages = state.get("messages", [])
        base_goal = state.get("base_goal")

        if not base_goal:
            # Find the last human message
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    base_goal = m.content
                    break

        if not base_goal:
            # No usable goal, just return state unchanged
            return state

        adjustments = state.get("adjustments", [])
        adjustment_count = len(adjustments)

        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            **state,
            "base_goal": base_goal,
            "current_goal": base_goal if not adjustments else " ".join([base_goal] + adjustments),
            "adjustments": adjustments,
            "adjustment_count": adjustment_count,
            "created_at": state.get("created_at", now_iso),
            "last_updated_at": now_iso,
            "done_collecting": False,
        }

    def _collect_adjustments(
        self, state: SequentialAdjustmentState
    ) -> SequentialAdjustmentState:
        """
        Node that uses `interrupt()` to gather adjustments or start a new task.

        This is where the official HITL pattern is used:
        - Graph stops at `interrupt(...)`
        - External caller resumes with `Command(resume="...")`
        """
        base_goal = state.get("base_goal", "").strip()
        current_goal = state.get("current_goal", base_goal)
        adjustments = state.get("adjustments", [])
        messages = state.get("messages", [])

        # Ask the human for an adjustment, a new task, or "done".
        prompt = (
            "Current task / combined goal:\n"
            f"  {current_goal}\n\n"
            "Enter one of the following:\n"
            "- Additional instructions to adjust/refine this task\n"
            "- A completely new task to replace it\n"
            "- 'done' (or just press Enter) when you're finished giving adjustments.\n"
        )
        user_text = interrupt(prompt)

        now_iso = datetime.now(timezone.utc).isoformat()

        # If user_text is None or empty => user is done giving adjustments
        if not user_text or not str(user_text).strip():
            return {
                **state,
                "done_collecting": True,
                "last_updated_at": now_iso,
            }

        user_text_str = str(user_text).strip()
        lower = user_text_str.lower()

        # If user explicitly says "done" / "no" / "continue"
        if lower in {"done", "no", "n", "ok", "okay", "continue", "go ahead"}:
            return {
                **state,
                "done_collecting": True,
                "last_updated_at": now_iso,
            }

        # Add this as a new human message in the history
        messages.append(HumanMessage(content=user_text_str))

        # Decide whether this is an adjustment or a completely new task
        if self._is_adjustment(user_text_str, current_goal or base_goal):
            # Adjustment case: append and recompute combined goal
            adjustments.append(user_text_str)
            combined_goal = " ".join([base_goal] + adjustments)

            print(f"[ADJUSTMENT #{len(adjustments)}] {user_text_str}")
            print(f"[UPDATED GOAL] {combined_goal}")

            return {
                **state,
                "messages": messages,
                "adjustments": adjustments,
                "adjustment_count": len(adjustments),
                "current_goal": combined_goal,
                "done_collecting": False,  # keep collecting until user says done
                "last_updated_at": now_iso,
            }
        else:
            # New task: reset adjustments and treat this as a fresh base_goal
            print(f"[NEW TASK] {user_text_str}")
            print("[RESET] Previous task is replaced by this new task.")

            return {
                **state,
                "messages": messages,
                "base_goal": user_text_str,
                "current_goal": user_text_str,
                "adjustments": [],
                "adjustment_count": 0,
                "done_collecting": False,
                "last_updated_at": now_iso,
            }

    def _route_after_collect(self, state: SequentialAdjustmentState) -> str:
        """Route logic after collect_adjustments."""
        if state.get("done_collecting", False):
            return "generate"
        return "more"

    def _generate_output(
        self, state: SequentialAdjustmentState
    ) -> SequentialAdjustmentState:
        """
        Final node: generate output using the combined goal and full message history.
        """
        base_goal = state.get("base_goal", "")
        current_goal = state.get("current_goal", base_goal)
        adjustments = state.get("adjustments", [])
        messages = state.get("messages", [])

        if not current_goal:
            return state

        print("\n[GENERATE] Creating final output using all requirements...")
        print(f"- Base goal: {base_goal}")
        if adjustments:
            print(f"- Adjustments ({len(adjustments)}):")
            for i, adj in enumerate(adjustments, start=1):
                print(f"  {i}. {adj}")

        if adjustments:
            adjustments_text = "\n".join(f"- {a}" for a in adjustments)
            final_prompt = f"""You are helping the user with an evolving task.

Initial user request:
\"\"\"{base_goal}\"\"\"

Additional requirements / adjustments provided later:
{adjustments_text}

Combined final task:
\"\"\"{current_goal}\"\"\"

Now produce a single, coherent final answer that:
- Fully satisfies the initial request, AND
- Incorporates **all** of the adjustments above.
"""
        else:
            final_prompt = f"""User request:
\"\"\"{current_goal}\"\"\"

Produce the best possible answer that follows these instructions.
"""

        # Use the entire conversation history for context, plus this final summarizing prompt.
        convo = messages + [HumanMessage(content=final_prompt)]

        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            response = self.llm.invoke(convo)
            output = response.content

            # Add assistant message to history as well
            messages.append(response)

            return {
                **state,
                "messages": messages,
                "output": output,
                "last_updated_at": now_iso,
            }
        except Exception as e:
            err_msg = f"Error generating output: {e}"
            print(err_msg)
            return {
                **state,
                "output": err_msg,
                "last_updated_at": now_iso,
            }

    # ------------------------------------------------------------------
    # CLI Runner using official `interrupt` + `Command(resume=...)` pattern
    # ------------------------------------------------------------------

    def run(self, thread_id: str = "sequential_adjustments_session"):
        """
        Run a single sequential-adjustment session via CLI.

        Pattern:
        - First call `graph.stream` with the initial user message.
        - The graph stops at `interrupt`.
        - While `state.next` is not empty, we resume with `Command(resume=...)`.
        - When finished, we inspect final state and print summary.
        """
        config = {"configurable": {"thread_id": thread_id}}

        print("=" * 80)
        print("Sequential Adjustment HITL Agent")
        print("=" * 80)
        print("Instructions:")
        print("- Enter any initial task (e.g., API design, spec writing, etc.)")
        print("- Then enter 0 or more adjustments.")
        print("- Type 'done' (or just press Enter) when you're finished adjusting.")
        print("- Type 'exit' or 'quit' at any prompt to stop.\n")

        initial = input("Initial task: ").strip()
        if not initial or initial.lower() in {"exit", "quit"}:
            print("Exiting.")
            return

        # Kick off the graph with the initial user message
        initial_messages = [HumanMessage(content=initial)]

        for _ in self.graph.stream(
            {"messages": initial_messages}, config, stream_mode="values"
        ):
            # We don't need per-event printing here; we're just driving the state machine.
            pass

        # Now handle any interrupts (multiple sequential adjustments)
        while True:
            state = self.graph.get_state(config)
            # If there is no "next", the graph has finished
            if not state.next:
                break

            # Graph is paused at interrupt() and expects human input
            user_input = input("\nAdjustment / new task / 'done': ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting.")
                return

            # Resume execution using LangGraph's official pattern
            for _ in self.graph.stream(
                Command(resume=user_input), config, stream_mode="values"
            ):
                pass

        # Final state
        final_state = self.graph.get_state(config).values

        base_goal = final_state.get("base_goal", "")
        current_goal = final_state.get("current_goal", "")
        adjustments = final_state.get("adjustments", []) or []
        adjustment_count = final_state.get("adjustment_count", 0)
        output = final_state.get("output", "")
        messages = final_state.get("messages", []) or []

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Base goal:          {base_goal}")
        print(f"Final combined goal:{current_goal}")
        print(f"Total adjustments:  {adjustment_count}")
        if adjustments:
            print("Adjustments (in order):")
            for idx, adj in enumerate(adjustments, 1):
                print(f"  {idx}. {adj}")

        print("\nSession History (user messages only):")
        print("-" * 80)
        i = 1
        for msg in messages:
            if isinstance(msg, HumanMessage):
                print(f"{i}. USER: {msg.content}")
                i += 1

        print("-" * 80)
        print("\nFinal Output:")
        print("-" * 80)
        print(output or "<no output generated>")
        print("-" * 80)


# --------------------------------------------------------------------------------------
# Script entrypoint
# --------------------------------------------------------------------------------------


def main():
    try:
        agent = SequentialAdjustmentAgent()
        agent.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
