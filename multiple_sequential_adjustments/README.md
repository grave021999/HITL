# Sequential Adjustment HITL Agent (LangGraph + LangChain)

This service implements a **Human-in-the-Loop (HITL)** agent using **LangGraph** and **LangChain** that supports **multiple sequential user adjustments in a single session**.

The agent is designed to cover scenarios like:

1. User: `Create a Python REST API for user management`  

2. Agent starts processing the task  

3. User: `Add JWT authentication`  

4. Agent updates the goal and continues  

5. User: `Use FastAPI framework specifically`  

6. Agent updates the goal again and continues  

7. Final output includes **all** adjustments  

8. Session history reflects **all** user inputs  

9. Agent maintains **conversation context** throughout  

> **Note:** The above is only an example. The agent works with **any initial goal** and **any sequence of follow-up messages**. No scenario is hardcoded.

---

## Overview

This component exposes a **stateful agent** that:

- Accepts an **initial goal** from the user.

- Enters a **processing phase** for that goal.

- During processing, repeatedly pauses using LangGraph's `interrupt()` to:

  - Accept **adjustments** (refinements to the current goal), or

  - Accept **new tasks** (which cancel and replace the current goal), or

  - Continue without change.

- Tracks:

  - The original goal

  - All adjustments

  - The combined "current goal"

  - Session history (all human messages)

- Generates a **final LLM output** that reflects:

  - The initial request

  - All accepted adjustments

- Implements the official **LangGraph HITL** pattern for `interrupt()` / `Command(resume=...)`.

This makes it suitable for production scenarios where:

- A user iteratively refines a request in a single conversation.

- You need **traceable state** (initial goal, adjustments, timestamps).

- You must demonstrate or enforce **Human-in-the-Loop** controls.

---

## Key Features

- ✅ **Multiple sequential adjustments in one session**  

  The user can send any number of follow-up messages while the agent is "working".

- ✅ **Adjustment vs. new-task classification**  

  A dedicated heuristic (`_is_adjustment`) decides whether a message:

  - Extends the current goal (**adjustment**) or

  - Cancels it and starts over (**new task**).

- ✅ **Context-preserving state management**  

  The agent keeps:

  - `messages` (full history)

  - `initial_goal`

  - `current_goal` (combined)

  - `all_adjustments`

  - `adjustment_count`

  - Timestamps (`task_started_at`, `last_adjustment_at`)

- ✅ **HITL via LangGraph `interrupt()`**  

  The agent pauses at well-defined points and resumes with `Command(resume=...)`, following the official pattern.

- ✅ **Scenario-agnostic**  

  No scenario string is hardcoded. Any initial goal and adjustment sequence can be used to validate behavior.

---

## Architecture

### State Schema

The agent uses a `SequentialAgentState` (`TypedDict`) aligned with LangGraph's `MessagesState` approach, extended with additional fields:

- `messages: list[BaseMessage]`  

  Full conversation history (initial goal + all follow-ups).

- `initial_goal: str`  

  First user request in the session.

- `current_goal: str`  

  Current combined goal (initial goal + all adjustments) or most recent new task.

- `all_adjustments: list[str]`  

  All user inputs that were classified as **adjustments**.

- `adjustment_count: int`  

  Number of adjustments.

- `output: str`  

  Final LLM output.

- `task_started_at: str`  

  ISO timestamp when the current task was started.

- `last_adjustment_at: Optional[str]`  

  ISO timestamp of the last adjustment.

- `continue_checking: bool`  

  Routing flag for whether to keep checking for further adjustments.

This state is persisted via `MemorySaver` (LangGraph checkpointer) to support pause/resume across `interrupt()` boundaries.

### Graph Topology

A `StateGraph[SequentialAgentState]` is built with three main nodes:

- `process`  

  Simulated "work" node which:

  - Periodically calls `interrupt()` during a 60-second processing window.

  - Accepts mid-flight adjustments or new tasks.

  - Updates state and routing flags accordingly.

- `check_input`  

  Node dedicated to:

  - Handling initial goal setup.

  - Asking the user for adjustments before/after processing chunks.

  - Classifying each new input as adjustment or new task.

- `generate`  

  Final node that:

  - Builds a combined prompt from `initial_goal` + `all_adjustments` + `current_goal`.

  - Calls the LLM with full conversation `messages` + combined prompt.

  - Stores the result in `state["output"]`.

**Entry point:** `process`  

**Checkpointer:** `MemorySaver()`  

**Threading:** A `thread_id` is used to bind all resumes of a session.

---

## Human-in-the-Loop Design

The implementation follows the official LangGraph HITL pattern:

- **Docs:**  

  **"How to wait for user input using `interrupt`"**  

  https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/

Core mechanics:

1. Inside graph nodes (`check_input`, `process`), the agent calls:

   ```python
   new_input = interrupt("Send adjustment or press Enter to continue.")
   ```

   This pauses the graph and returns control to the caller (CLI, server, UI, etc.).

2. When new user input is available, the graph is resumed with:

   ```python
   Command(resume=user_input)
   ```

3. The agent classifies `user_input` via `_is_adjustment()`:

   - **Adjustment** → Append to `all_adjustments`, rebuild `current_goal`, update timestamps.

   - **New task** → Reset `initial_goal`, `current_goal`, `all_adjustments`, `adjustment_count`, and timestamps.

4. The graph then routes:

   - Back to `process` to continue working with the updated goal, or

   - On to `generate` when no more adjustments are pending and processing is complete.

This pattern is fully compatible with production UIs (CLI, Slack, web frontends) that can drive the `Command(resume=...)` loop.

---

## Installation & Configuration

### Requirements

- Python ≥ 3.10

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
python SequentialAdjustmentAgent.py
```

The CLI will:

- Prompt for the initial task.

- Start a processing phase for ~60 seconds (simulated).

- During processing, the agent will pause multiple times (via `interrupt()`) and prompt:

  - If you enter a message → it is treated as adjustment or new task.

  - If you press Enter → the agent continues working unchanged.

- After processing, it will:

  - Generate final output.

  - Print a structured **FINAL RESULTS** section:

    - Initial Goal

    - Total Adjustments

    - List of adjustments

    - Final Combined Goal

    - Session History (all user inputs)

    - Output length + final content

---

## Example: Testing the Multiple Sequential Adjustments Scenario

You can validate your target scenario end-to-end without any hardcoding.

**Start the agent:**

```bash
python SequentialAdjustmentAgent.py
```

**At the initial prompt, enter:**

```
Create a Python REST API for user management
```

**During processing, at the first or second interrupt prompt, enter:**

```
Add JWT authentication
```

**At the next interrupt prompt, enter:**

```
Use FastAPI framework specifically
```

**Allow processing to complete and wait for the final summary.**

**Verify:**

- Initial Goal = "Create a Python REST API for user management"

- Total Adjustments = 2

- Adjustments:
  - Add JWT authentication
  - Use FastAPI framework specifically

- Final Combined Goal contains all of the above.

- Session History lists all three user messages.

- Final Output reflects:
  - Python REST API
  - User management
  - JWT authentication
  - FastAPI as the framework

This directly matches the "Multiple Sequential Adjustments" requirement while still allowing you to plug in any other scenario for validation.

---

## Behavior for Arbitrary Scenarios (Not Hardcoded)

The `_is_adjustment` heuristic ensures the logic is scenario-agnostic:

- Short, additive-sounding messages (e.g., "Add JWT auth", "Use FastAPI", "Also make it paginated") tend to be treated as **adjustments**.

- Inputs that clearly switch topic or cancel (e.g., "Actually, just focus on quantum entanglement, make it 2 pages") are treated as **new tasks**, resetting the current goal.

Therefore you can test:

- API design scenarios

- Document generation scenarios

- Planning / summarization tasks

- Any other domain-specific workflows

without changing the core agent code.

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

The additional logic (adjustment detection, timestamps, reporting) is layered on top of these official patterns to satisfy the "Multiple Sequential Adjustments in a Single Session" production requirement.
