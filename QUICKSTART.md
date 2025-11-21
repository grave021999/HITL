# Quick Start Guide

## Setup (1 minute)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your API key:**
   Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Run the Test

```bash
python hitl_test.py
```

This will:
- Start with task: "Write a comprehensive 10-page report on quantum computing"
- Wait 60 seconds (simulating work in progress)
- Send interrupt: "Actually, just focus on quantum entanglement, make it 2 pages"
- Show that the original task is cancelled and new task is adopted

The test includes detailed assertions to verify:
- ✓ Previous task is cancelled
- ✓ New task direction is adopted
- ✓ Original report is not completed
- ✓ Output matches new requirements

## How It Works

The agent uses LangGraph's state management to:
1. Track the current goal in state
2. Detect when a new human message arrives
3. Compare new message with current goal
4. Cancel previous task if goals differ
5. Replan and regenerate based on new goal

The key is the `check_new_input` node that runs before planning/generation to detect interrupts.

