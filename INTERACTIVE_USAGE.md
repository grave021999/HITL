# Interactive HITL Agent - Usage Guide

## Overview

This interactive agent allows you to:
- Enter **any query/task** (not hardcoded)
- Agent processes for 60 seconds
- **During processing**, you can send a **NEW query** to interrupt
- New query **cancels** previous task and **switches immediately**
- Works for **any topic**, not just quantum computing

## How It Works

1. **Start the agent**: `python HITL_Agent.py`
2. **Enter your first query**: Type any task (e.g., "Write a report on AI")
3. **Agent starts processing**: Works for 60 seconds
4. **During processing**: 
   - Every 10 seconds, the agent pauses and asks if you want to interrupt
   - You can press **Enter** to continue, or
   - Type a **NEW query** to interrupt and cancel the current task
5. **If interrupted**: Previous task is cancelled, new task starts immediately
6. **Final output**: Generated based on the latest task

## Example Session

```
Enter your first task/query: Write a comprehensive 10-page report on quantum computing

[START] Task: Write a comprehensive 10-page report on quantum computing
[PROCESS] Starting to work on: Write a comprehensive 10-page report on quantum computing
[INFO] You can send a new query anytime during the next 60 seconds to interrupt!
[WORK] Processing... (0s / 60s elapsed)

> Actually, just focus on quantum entanglement, make it 2 pages

[INTERRUPT] New task received during work: 'Actually, just focus on quantum entanglement, make it 2 pages'
[CANCEL] Stopping current work on: 'Write a comprehensive 10-page report on quantum computing'
[GENERATE] Generating output for: Actually, just focus on quantum entanglement, make it 2 pages

FINAL RESULTS
Previous Goal: Write a comprehensive 10-page report on quantum computing
Current Goal: Actually, just focus on quantum entanglement, make it 2 pages
Cancelled: True
```

## Key Features

✅ **No hardcoded inputs** - Enter any query  
✅ **Real-time interruption** - Send new queries during processing  
✅ **Task cancellation** - Previous task is cancelled when interrupted  
✅ **Immediate switching** - New task starts right away  
✅ **Works for any topic** - Not limited to specific subjects  

## Requirements

- Python 3.8+
- `ANTHROPIC_API_KEY` in environment or `.env` file
- Required packages: `langgraph`, `langchain-anthropic`, `python-dotenv`

## Notes

- The agent checks for new input every 10 seconds during the 60-second processing window
- If you press Enter (empty input), the agent continues with the current task
- If you send a new query, it immediately cancels and switches
- Type 'exit' or 'quit' to stop the session

