# Interactive Human-in-the-Loop (HITL) Agent

An interactive agent that allows unsolicited user input to interrupt and cancel ongoing tasks, following the official LangGraph interrupt pattern.

## Features

- ✅ **Interactive terminal interface** - Enter any query/task
- ✅ **Real-time interruption** - Send new queries during 60-second processing window
- ✅ **Task cancellation** - Previous task is cancelled when interrupted
- ✅ **Immediate switching** - New task starts right away
- ✅ **Works for any topic** - Not limited to specific subjects
- ✅ **Follows LangGraph documentation** - Uses official `interrupt()` pattern

## Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key

### Installation

1. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

2. Set your Anthropic API key by creating a `.env` file:
```
ANTHROPIC_API_KEY=your_api_key_here
```

Or set it as an environment variable:
```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

### Usage

Run the interactive agent:
```bash
python HITL_Agent.py
```

## How It Works

1. **Enter your first query** - Type any task (e.g., "Write a report on AI")
2. **Agent starts processing** - Works for 60 seconds
3. **During processing**:
   - Every 10 seconds, the agent pauses and asks if you want to interrupt
   - Press **Enter** to continue, or
   - Type a **NEW query** to interrupt and cancel the current task
4. **If interrupted**: Previous task is cancelled, new task starts immediately
5. **Final output**: Generated based on the latest task

## Example Session

```
Enter your first task/query: Write a comprehensive 10-page report on quantum computing

[PROCESS] Starting to work on: Write a comprehensive 10-page report on quantum computing
[WORK] Processing... (0s / 60s elapsed)

> Actually, just focus on quantum entanglement, make it 2 pages

[INTERRUPT] New task received during work: 'Actually, just focus on quantum entanglement, make it 2 pages'
[CANCEL] Stopping current work on: 'Write a comprehensive 10-page report on quantum computing'
[GENERATE] Generating output for: Actually, just focus on quantum entanglement, make it 2 pages
```

## Implementation Details

This implementation follows the official LangGraph HITL pattern:
- Uses `interrupt()` function calls inside nodes (dynamic interrupts)
- Uses `Command(resume=value)` to resume execution
- Uses `MemorySaver` checkpointer for state persistence
- Follows TypedDict state schema pattern

See [INTERACTIVE_USAGE.md](INTERACTIVE_USAGE.md) for detailed usage guide.

## Documentation References

- [LangGraph Interrupts Documentation](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [Wait for User Input Guide](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
