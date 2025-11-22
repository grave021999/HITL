# Multiple Sequential Adjustments - HITL Agent

An agent that handles multiple sequential user corrections during a single session, following the official LangGraph HITL interrupt pattern.

## Features

- ✅ **Multiple sequential adjustments** - Handle multiple corrections in one session
- ✅ **Combined output** - Final output includes all adjustments
- ✅ **Session history** - All user inputs are tracked and maintained
- ✅ **Conversation context** - Agent maintains context across all adjustments
- ✅ **Follows LangGraph documentation** - Uses official `interrupt()` pattern
- ✅ **Interactive** - No hardcoded scenarios, works with any input

## Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key

### Installation

1. Install dependencies (from parent directory):
```bash
python -m pip install -r ../requirements.txt
```

2. Set your Anthropic API key in `.env` file (in parent directory):
```
ANTHROPIC_API_KEY=your_api_key_here
```

### Usage

Run the sequential adjustment agent:
```bash
python SequentialAdjustmentAgent.py
```

## How It Works

1. **Enter initial task** - Type your first task/query
2. **Agent starts processing** - Works for 60 seconds
3. **Send adjustments** - During processing, send multiple sequential adjustments
4. **All adjustments combined** - Each adjustment is added to the goal
5. **Final output** - Generated output includes all adjustments
6. **Session history** - All inputs are tracked and displayed

## Example Session

```
Enter your initial task/query: Create a Python REST API for user management

[START] Initial task: Create a Python REST API for user management
[PROCESS] Starting to work on: Create a Python REST API for user management

> Add JWT authentication

[ADJUSTMENT #1] Received: 'Add JWT authentication'
[UPDATE] Combined goal: 'Create a Python REST API for user management Add JWT authentication'

> Use FastAPI framework specifically

[ADJUSTMENT #2] Received: 'Use FastAPI framework specifically'
[UPDATE] Combined goal: 'Create a Python REST API for user management Add JWT authentication Use FastAPI framework specifically'

[GENERATE] Generating output with all adjustments...
  Initial goal: Create a Python REST API for user management
  Adjustments (2): Add JWT authentication, Use FastAPI framework specifically

FINAL RESULTS
Initial Goal: Create a Python REST API for user management
Total Adjustments: 2
Adjustments:
  1. Add JWT authentication
  2. Use FastAPI framework specifically
Final Combined Goal: Create a Python REST API for user management Add JWT authentication Use FastAPI framework specifically
```

## Verification Requirements

The agent verifies:

1. ✅ **Final output includes all adjustments** - All sequential adjustments are reflected in output
2. ✅ **Session history reflects all user inputs** - All inputs are tracked and displayed
3. ✅ **Agent maintains conversation context** - Context is maintained across all adjustments

## Implementation Details

This implementation follows the official LangGraph HITL pattern:
- Uses `interrupt()` function calls inside nodes (dynamic interrupts)
- Uses `Command(resume=value)` to resume execution
- Uses `MemorySaver` checkpointer for state persistence
- Follows TypedDict state schema pattern
- Maintains conversation history in messages

## Documentation References

- [LangGraph Interrupts Documentation](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Wait for User Input Guide](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)

