# Implementation Details

## Architecture

The HITL (Human-in-the-Loop) implementation uses LangGraph to create an agent that can be interrupted and redirected during execution.

### Key Components

1. **AgentState** - Tracks:
   - Messages (conversation history)
   - Current goal
   - Previous goal (for cancellation tracking)
   - Cancelled flag
   - Output
   - Timestamps

2. **Graph Nodes**:
   - `check_new_input` - Entry point that detects new user messages
   - `plan` - Plans the task based on current goal
   - `generate` - Generates content using LLM

3. **Interrupt Flow**:
   ```
   check_new_input → (detects new message) → plan → generate
                      (if cancelled) → END
   ```

## How Interruption Works

1. **Initial State**: Agent receives first message, sets `current_goal`
2. **Planning Phase**: Agent plans the task
3. **Work Simulation**: Agent simulates working (waits N seconds)
4. **Interrupt Arrives**: New message is added to state
5. **Detection**: `check_new_input` compares latest message with `current_goal`
6. **Cancellation**: If different, sets `cancelled=True`, stores previous goal
7. **Replanning**: Agent replans with new goal
8. **Regeneration**: Agent generates content for new goal

## State Management

Uses LangGraph's `MemorySaver` checkpointer to:
- Persist state across graph executions
- Track conversation history
- Enable state updates mid-execution

## Test Scenario

The test verifies:
1. ✓ Task cancellation when new input arrives
2. ✓ New goal adoption
3. ✓ Original task not completed
4. ✓ Output matches new requirements

## Usage Pattern

```python
agent = ReportAgent(api_key=api_key)
result = await agent.process_with_interrupt(
    thread_id="unique_id",
    initial_message="Original task",
    interrupt_after_seconds=60.0,
    interrupt_message="New task"
)
```

## Key Design Decisions

1. **State-based cancellation**: Uses state flags rather than exceptions
2. **Message comparison**: Detects interrupts by comparing message content
3. **Replanning on interrupt**: Always replans when new input detected
4. **Simulated work**: Uses `asyncio.sleep` to simulate long-running tasks

## Future Enhancements

- Use LangGraph's built-in interrupt mechanisms (`interrupt_before`/`interrupt_after`)
- Add more sophisticated task comparison (semantic similarity)
- Support partial task completion before interrupt
- Add progress tracking


