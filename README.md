# HITL Agent (LangGraph + LangChain)

This service implements a **Human-in-the-Loop (HITL)** agent using **LangGraph** and **LangChain** that supports **unsolicited user input to interrupt and cancel ongoing tasks**.

The agent is designed to cover scenarios like:

1. User: `Write a comprehensive 10-page report on quantum computing`  

2. Agent starts processing the task  

3. User interrupts: `Actually, just focus on quantum entanglement, make it 2 pages`  

4. Agent **cancels** the previous task and **switches** to the new task immediately  

5. Final output reflects **only** the latest task  

> **Note:** The above is only an example. The agent works with **any initial goal** and **any interrupting message**. No scenario is hardcoded.

---

## Overview

This component exposes a **stateful agent** that:

- Accepts an **initial goal** from the user.

- Enters a **processing phase** for that goal (simulated 60-second work period).

- During processing, repeatedly pauses using LangGraph's `interrupt()` to:

  - Accept **follow-up adjustments** (refinements that extend the current goal), or

  - Accept **new tasks** (which cancel and replace the current goal), or

  - Continue without change.

- Tracks:

  - The current goal

  - The previous goal (if cancelled)

  - Cancellation status and timestamps

  - Session history (all human messages)

- Generates a **final LLM output** that reflects:

  - The latest request only (if task was cancelled, previous work is discarded)

- Implements the official **LangGraph HITL** pattern for `interrupt()` / `Command(resume=...)`.

This makes it suitable for production scenarios where:

- A user needs to **cancel** an ongoing task and redirect the agent.

- You need **traceable state** (current goal, previous goal, cancellation timestamps).

- You must demonstrate or enforce **Human-in-the-Loop** controls for task redirection.

---

## Key Features

- ✅ **Unsolicited interruption during processing**  

  The user can send interrupting messages at any point during the 60-second processing window.

- ✅ **Follow-up vs. new-task classification**  

  A dedicated heuristic (`_is_follow_up`) decides whether a message:

  - Extends the current goal (**follow-up adjustment**) or

  - Cancels it and starts over (**new task**).

- ✅ **Task cancellation and redirection**  

  When a new task is detected:

  - Previous goal is stored in `previous_goal`.

  - Current goal is replaced with the new task.

  - `cancelled` flag is set to `True`.

  - Timestamp is recorded in `task_cancelled_at`.

- ✅ **Context-preserving state management**  

  The agent keeps:

  - `messages` (full history)

  - `current_goal`

  - `previous_goal` (if cancelled)

  - `cancelled` (boolean flag)

  - Timestamps (`task_started_at`, `task_cancelled_at`)

- ✅ **HITL via LangGraph `interrupt()`**  

  The agent pauses at well-defined points and resumes with `Command(resume=...)`, following the official pattern.

- ✅ **Scenario-agnostic**  

  No scenario string is hardcoded. Any initial goal and interrupting message can be used to validate behavior.

---

## Architecture

### State Schema

The agent uses an `AgentState` (`TypedDict`) aligned with LangGraph's `MessagesState` approach, extended with additional fields:

- `messages: list[BaseMessage]`  

  Full conversation history (initial goal + all follow-ups/interruptions).

- `current_goal: str`  

  Current goal (may be combined with follow-ups, or replaced by new task).

- `previous_goal: str`  

  Previous goal if it was cancelled by a new task.

- `cancelled: bool`  

  Whether the current task was cancelled (replaced by a new task).

- `output: str`  

  Final LLM output.

- `task_started_at: str`  

  ISO timestamp when the current task was started.

- `task_cancelled_at: Optional[str]`  

  ISO timestamp when the task was cancelled (if applicable).

This state is persisted via `MemorySaver` (LangGraph checkpointer) to support pause/resume across `interrupt()` boundaries.

### Graph Topology

A `StateGraph[AgentState]` is built with three main nodes:

- `check_input`  

  Node that:

  - Handles initial goal setup.

  - Calls `interrupt()` to check for new input before processing.

  - Classifies each new input as follow-up or new task.

  - Updates state accordingly (combine for follow-up, cancel for new task).

- `process`  

  Simulated "work" node which:

  - Processes for 60 seconds (simulated with `time.sleep`).

  - Periodically calls `interrupt()` every 10 seconds during the processing window.

  - Accepts mid-flight follow-ups or new tasks.

  - Updates state and routing flags accordingly.

- `generate`  

  Final node that:

  - Builds a prompt from `current_goal` (latest task only).

  - Calls the LLM with the prompt.

  - Stores the result in `state["output"]`.

  - **Note:** If task was cancelled, output reflects only the new task, not the previous one.

**Entry point:** `check_input`  

**Checkpointer:** `MemorySaver()`  

**Threading:** A `thread_id` is used to bind all resumes of a session.

**Routing:** Conditional edges from `check_input` route to:
- `process` (if continuing with current task)
- `check_input` (if new task received, loop back to re-check)

---

## Human-in-the-Loop Design

The implementation follows the official LangGraph HITL pattern:

- **Docs:**  

  **"How to wait for user input using `interrupt`"**  

  https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/

Core mechanics:

1. Inside graph nodes (`check_input`, `process`), the agent calls:

   ```python
   new_input = interrupt("Send new task to interrupt, or press Enter to continue.")
   ```

   This pauses the graph and returns control to the caller (CLI, server, UI, etc.).

2. When new user input is available, the graph is resumed with:

   ```python
   Command(resume=user_input)
   ```

3. The agent classifies `user_input` via `_is_follow_up()`:

   - **Follow-up** → Combine with `current_goal`, keep `cancelled = False`.

   - **New task** → Store `current_goal` in `previous_goal`, set `current_goal = new_input`, set `cancelled = True`, record timestamp.

4. The graph then routes:

   - To `process` to continue working with the updated goal, or

   - Back to `check_input` if a new task was received (to re-check), or

   - On to `generate` when processing is complete.

This pattern is fully compatible with production UIs (CLI, Slack, web frontends) that can drive the `Command(resume=...)` loop.

---

## Installation & Configuration

### Requirements

- Python ≥ 3.8

- `langgraph`

- `langchain-core`

- `langchain-anthropic`

- `python-dotenv`

### Environment Variables

Configure Anthropic via `.env` or environment variables:

```bash
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-3-5-haiku-20241022  # or another Anthropic model
```

### Install Dependencies

```bash
pip install langgraph langchain-core langchain-anthropic python-dotenv
```

or, if you have a requirements.txt:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Agent (CLI Mode)

From the project root:

```bash
python HITL_Agent.py
```

The CLI will:

- Prompt for the initial task.

- Start a processing phase for ~60 seconds (simulated).

- During processing, the agent will pause multiple times (via `interrupt()`) every 10 seconds and prompt:

  - If you enter a message → it is classified as follow-up or new task.

  - If you press Enter → the agent continues working unchanged.

- After processing, it will:

  - Generate final output.

  - Print a structured **FINAL RESULTS** section:

    - Previous Goal (if cancelled)

    - Current Goal

    - Cancelled status

    - Output length + final content

---

## Example: Testing the Task Cancellation Scenario

You can validate your target scenario end-to-end without any hardcoding.

**Start the agent:**

```bash
python HITL_Agent.py
```

**At the initial prompt, enter:**

```
Write a comprehensive 10-page report on quantum computing
```

**During processing, at the first or second interrupt prompt, enter:**

```
Actually, just focus on quantum entanglement, make it 2 pages
```

**Allow processing to complete and wait for the final summary.**

**Verify:**

- Previous Goal = "Write a comprehensive 10-page report on quantum computing"

- Current Goal = "Actually, just focus on quantum entanglement, make it 2 pages"

- Cancelled = `True`

- Final Output reflects:
  - Quantum entanglement (new focus)
  - 2 pages (new requirement)
  - **Not** a 10-page comprehensive report (previous task was cancelled)

This directly matches the "Task Cancellation and Redirection" requirement while still allowing you to plug in any other scenario for validation.

---

## Behavior for Arbitrary Scenarios (Not Hardcoded)

The `_is_follow_up` heuristic ensures the logic is scenario-agnostic:

- Short, additive-sounding messages (e.g., "Add JWT auth", "Also make it paginated", "Use FastAPI") tend to be treated as **follow-ups** (combined with current goal).

- Inputs that clearly switch topic or cancel (e.g., "Actually, just focus on quantum entanglement, make it 2 pages", "Forget that, write about AI instead") are treated as **new tasks**, cancelling the previous goal.

Therefore you can test:

- Task cancellation scenarios (quantum computing → quantum entanglement)

- Task redirection scenarios (any topic switch)

- Follow-up refinement scenarios (additions to current task)

- Any other domain-specific workflows

without changing the core agent code.

---

## Key Differences from Sequential Adjustment Agent

This agent focuses on **task cancellation** rather than **sequential adjustments**:

| Feature | HITL Agent | Sequential Adjustment Agent |
|---------|-----------|----------------------------|
| **Primary Use Case** | Cancel and redirect tasks | Combine multiple adjustments |
| **Follow-up Behavior** | Can combine OR cancel | Always combines |
| **New Task Behavior** | Cancels previous, replaces | Cancels previous, resets |
| **Output Focus** | Latest task only | All adjustments combined |
| **State Tracking** | `previous_goal`, `cancelled` | `initial_goal`, `all_adjustments` |

Both agents use the same LangGraph HITL pattern but serve different use cases:

- **HITL Agent**: When you need to **cancel** and **redirect** the agent's work.

- **Sequential Adjustment Agent**: When you need to **iteratively refine** a request with multiple adjustments.

---

## Alignment with Official Documentation

The implementation is guided by:

**LangGraph Human-in-the-Loop:**

- **How to wait for user input using interrupt**

  https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/

**LangChain / LangGraph main docs:**

- LangChain docs: https://docs.langchain.com/

- LangGraph docs: https://langchain-ai.github.io/langgraph/

From these docs, this agent adopts:

- `StateGraph` + `MemorySaver` for stateful, resumable execution.

- `interrupt()` and `Command(resume=...)` for controlled Human-in-the-Loop interactions.

- A messages-based state (`messages: list[BaseMessage]`) for consistent context.

The additional logic (follow-up detection, cancellation tracking, timestamps) is layered on top of these official patterns to satisfy the "Unsolicited User Input Cancels Ongoing Tasks" production requirement.
