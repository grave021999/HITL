# Web UI for Sequential Adjustment HITL Agent

A web-based interface that implements **continuous background processing** with real-time adjustment handling.

## Features

✅ **Continuous Processing** - Processing runs for 60 seconds in the background  
✅ **Real-time Updates** - Progress updates sent to UI via WebSocket  
✅ **Non-blocking Adjustments** - Send adjustments at any time without interrupting processing  
✅ **Visual Feedback** - See progress, adjustments, and output in real-time  
✅ **No Enter Key Required** - Unlike terminal version, processing continues automatically  

## How It Works

1. **Background Thread** - Processing runs continuously in a separate thread
2. **Adjustment Queue** - UI sends adjustments to a queue that processing checks periodically
3. **WebSocket Communication** - Real-time bidirectional communication between UI and server
4. **Non-blocking** - Processing continues while checking for adjustments every 2 seconds

## Installation

1. Install web dependencies:
```bash
pip install -r requirements_web.txt
```

2. Make sure you have your `.env` file with `ANTHROPIC_API_KEY`

## Usage

1. Start the web server:
```bash
python web_ui_server.py
```

2. Open your browser and go to:
```
http://localhost:8000
```

3. **Enter initial task** - Type your query and click "Start Processing"

4. **Processing starts immediately** - Runs for 60 seconds continuously

5. **Send adjustments anytime** - Use the adjustment input field to send:
   - **Adjustments**: "Add JWT authentication", "Use FastAPI framework"
   - **New tasks**: "Actually, just focus on quantum entanglement"

6. **Watch progress** - See real-time updates, adjustments, and final output

## Key Differences from Terminal Version

| Feature | Terminal Version | Web UI Version |
|---------|-----------------|----------------|
| Processing | Blocks at interrupts | Runs continuously |
| User Input | Press Enter every 10s | Automatic, no input needed |
| Adjustments | Must wait for interrupt | Send anytime |
| Progress | Text-based | Visual with real-time updates |

## Architecture

```
┌─────────────┐
│   Browser   │
│   (React)   │
└──────┬──────┘
       │ WebSocket
       │
┌──────▼──────────────────┐
│   FastAPI Server        │
│   ┌──────────────────┐  │
│   │ Background Thread │  │
│   │ (Processing)     │  │
│   └────────┬─────────┘  │
│            │             │
│   ┌────────▼─────────┐  │
│   │ Adjustment Queue │  │
│   └─────────────────┘  │
└─────────────────────────┘
```

## Test Cases Supported

### Test Case 1: Unsolicited Input (Cancellation)
- Start: "Write a comprehensive 10-page report on quantum computing"
- During processing: "Actually, just focus on quantum entanglement, make it 2 pages"
- Result: Previous task cancelled, new task adopted

### Test Case 2: Multiple Sequential Adjustments
- Start: "Create a Python REST API for user management"
- Adjustment 1: "Add JWT authentication"
- Adjustment 2: "Use FastAPI framework specifically"
- Result: All adjustments combined in final output

## API Endpoints

- `GET /` - Web UI interface
- `WebSocket /ws` - Real-time communication

## WebSocket Messages

**Client → Server:**
```json
{"type": "start", "goal": "Your initial task"}
{"type": "adjustment", "text": "Add JWT authentication"}
```

**Server → Client:**
```json
{"type": "start", "message": "Starting to work on: ..."}
{"type": "progress", "elapsed": 10, "remaining": 50, "message": "..."}
{"type": "adjustment", "adjustment": "...", "count": 1, "combined_goal": "..."}
{"type": "cancellation", "new_task": "...", "message": "..."}
{"type": "complete", "message": "Finished processing"}
{"type": "output", "output": "Generated content..."}
```

