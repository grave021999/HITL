"""
Web UI Server for Sequential Adjustment HITL Agent

This implements continuous processing with a separate mechanism to check for adjustments.
Processing runs in the background while the UI can send adjustments at any time.

Usage:
    python web_ui_server.py

Then open http://localhost:8000 in your browser
"""

import asyncio
import os
import time
import threading
import json
from datetime import datetime
from typing import Optional, TypedDict, Annotated, List, Dict
from queue import Queue, Empty

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class SequentialAgentState(TypedDict):
    """State schema following LangGraph documentation pattern."""
    messages: Annotated[list[BaseMessage], add_messages]
    initial_goal: str
    current_goal: str
    all_adjustments: List[str]
    adjustment_count: int
    output: str
    task_started_at: str
    last_adjustment_at: Optional[str]
    continue_checking: bool
    processing_complete: bool


class WebUISequentialAgent:
    """
    Agent that handles multiple sequential adjustments with continuous background processing.
    Uses a separate thread for processing while UI can send adjustments at any time.
    """
    
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
        
        # Shared state for UI communication
        self.adjustment_queue = Queue()  # Queue for adjustments from UI
        self.progress_updates = Queue()  # Queue for progress updates to UI
        self.processing_active = False
        self.current_state = None
        self.config = None
        
    def _is_adjustment(self, new_input: str, current_goal: str) -> bool:
        """Determine if new input is an adjustment or a new task."""
        new_input_lower = new_input.strip().lower()
        
        # Cancellation indicators
        cancellation_indicators = [
            "actually", "instead", "just focus on", "just do", "forget", "ignore",
            "cancel", "stop", "change to", "switch to", "do this instead"
        ]
        
        for indicator in cancellation_indicators:
            if new_input_lower.startswith(indicator):
                return False
        
        # Adjustment indicators
        adjustment_indicators = [
            "also", "and", "add", "include", "make it", "update", "modify",
            "use", "specifically", "with", "implement", "ensure"
        ]
        
        for indicator in adjustment_indicators:
            if new_input_lower.startswith(indicator):
                return True
        
        # Default: treat as adjustment if short
        return len(new_input.strip().split()) <= 10
    
    def _process_continuously(self, initial_goal: str, thread_id: str):
        """
        Process continuously for 60 seconds in background thread.
        Checks for adjustments from UI queue periodically.
        """
        config = {"configurable": {"thread_id": thread_id}}
        self.config = config
        
        # Initialize state
        initial_state: SequentialAgentState = {
            "messages": [HumanMessage(content=initial_goal)],
            "initial_goal": initial_goal,
            "current_goal": initial_goal,
            "all_adjustments": [],
            "adjustment_count": 0,
            "output": "",
            "task_started_at": datetime.now().isoformat(),
            "last_adjustment_at": None,
            "continue_checking": False,
            "processing_complete": False,
        }
        
        self.current_state = initial_state
        goal = initial_goal
        work_duration = 60.0
        start_time = time.time()
        check_interval = 2.0  # Check for adjustments every 2 seconds
        
        self.processing_active = True
        self.progress_updates.put({
            "type": "start",
            "message": f"Starting to work on: {goal}",
            "goal": goal
        })
        
        while self.processing_active:
            elapsed = time.time() - start_time
            remaining = work_duration - elapsed
            
            if remaining <= 0:
                break
            
            # Send progress update
            self.progress_updates.put({
                "type": "progress",
                "elapsed": int(elapsed),
                "remaining": int(remaining),
                "message": f"Processing... ({int(elapsed)}s / {int(work_duration)}s elapsed)"
            })
            
            # Check for adjustments from UI (non-blocking)
            try:
                adjustment = self.adjustment_queue.get_nowait()
                if adjustment:
                    is_adjustment = self._is_adjustment(adjustment, goal)
                    
                    if is_adjustment:
                        # Sequential adjustment
                        all_adjustments = self.current_state.get("all_adjustments", [])
                        all_adjustments.append(adjustment)
                        
                        initial_goal = self.current_state.get("initial_goal", goal)
                        combined_parts = [initial_goal] + all_adjustments
                        combined_goal = " ".join(combined_parts)
                        
                        self.current_state["current_goal"] = combined_goal
                        self.current_state["all_adjustments"] = all_adjustments
                        self.current_state["adjustment_count"] = len(all_adjustments)
                        self.current_state["last_adjustment_at"] = datetime.now().isoformat()
                        self.current_state["messages"].append(HumanMessage(content=adjustment))
                        
                        goal = combined_goal
                        
                        self.progress_updates.put({
                            "type": "adjustment",
                            "adjustment": adjustment,
                            "count": len(all_adjustments),
                            "combined_goal": combined_goal,
                            "message": f"Adjustment #{len(all_adjustments)} received: '{adjustment}'"
                        })
                    else:
                        # New task - cancel previous
                        self.current_state["initial_goal"] = adjustment
                        self.current_state["current_goal"] = adjustment
                        self.current_state["all_adjustments"] = []
                        self.current_state["adjustment_count"] = 0
                        self.current_state["task_started_at"] = datetime.now().isoformat()
                        self.current_state["messages"].append(HumanMessage(content=adjustment))
                        
                        goal = adjustment
                        start_time = time.time()  # Reset timer for new task
                        
                        self.progress_updates.put({
                            "type": "cancellation",
                            "new_task": adjustment,
                            "message": f"Previous task cancelled. New task: '{adjustment}'"
                        })
            except Empty:
                pass  # No adjustment received, continue processing
            
            # Sleep for check interval
            time.sleep(check_interval)
        
        # Processing complete
        self.progress_updates.put({
            "type": "complete",
            "message": f"Finished processing: {goal}",
            "goal": goal
        })
        
        # Generate output
        self._generate_output(goal)
        
        self.processing_active = False
    
    def _generate_output(self, goal: str):
        """Generate final output based on combined goal."""
        initial_goal = self.current_state.get("initial_goal", goal)
        all_adjustments = self.current_state.get("all_adjustments", [])
        messages = self.current_state.get("messages", [])
        
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
            context_messages = messages + [HumanMessage(content=prompt)]
            response = self.llm.invoke(context_messages)
            
            self.current_state["output"] = response.content
            
            self.progress_updates.put({
                "type": "output",
                "output": response.content,
                "message": "Output generated successfully"
            })
        except Exception as e:
            self.progress_updates.put({
                "type": "error",
                "error": str(e),
                "message": f"Error generating output: {e}"
            })
    
    def start_processing(self, initial_goal: str, thread_id: str = "web_ui_session"):
        """Start background processing thread."""
        if self.processing_active:
            return False
        
        thread = threading.Thread(
            target=self._process_continuously,
            args=(initial_goal, thread_id),
            daemon=True
        )
        thread.start()
        return True
    
    def send_adjustment(self, adjustment: str):
        """Send adjustment to processing thread."""
        if self.processing_active:
            self.adjustment_queue.put(adjustment)
            return True
        return False
    
    def stop_processing(self):
        """Stop processing."""
        self.processing_active = False


# Global agent instance
agent = WebUISequentialAgent()


@app.get("/")
async def get_ui():
    """Serve the web UI."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sequential Adjustment HITL Agent</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            .input-section {
                margin: 20px 0;
            }
            input[type="text"] {
                width: 70%;
                padding: 12px;
                font-size: 16px;
                border: 2px solid #ddd;
                border-radius: 5px;
            }
            button {
                padding: 12px 24px;
                font-size: 16px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-left: 10px;
            }
            button:hover {
                background: #45a049;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .status {
                margin: 20px 0;
                padding: 15px;
                background: #e8f5e9;
                border-left: 4px solid #4CAF50;
                border-radius: 5px;
            }
            .progress {
                margin: 20px 0;
                padding: 15px;
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                border-radius: 5px;
            }
            .adjustment {
                margin: 10px 0;
                padding: 10px;
                background: #e3f2fd;
                border-left: 4px solid #2196F3;
                border-radius: 5px;
            }
            .output {
                margin: 20px 0;
                padding: 15px;
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                white-space: pre-wrap;
                max-height: 500px;
                overflow-y: auto;
            }
            .history {
                margin: 20px 0;
                padding: 15px;
                background: #f9f9f9;
                border-radius: 5px;
            }
            .history-item {
                margin: 5px 0;
                padding: 5px;
                background: white;
                border-left: 3px solid #4CAF50;
                padding-left: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sequential Adjustment HITL Agent</h1>
            
            <div class="input-section">
                <input type="text" id="initialInput" placeholder="Enter your initial task/query">
                <button onclick="startProcessing()" id="startBtn">Start Processing</button>
            </div>
            
            <div class="input-section">
                <input type="text" id="adjustmentInput" placeholder="Send adjustment or new task during processing" disabled>
                <button onclick="sendAdjustment()" id="adjustBtn" disabled>Send Adjustment</button>
            </div>
            
            <div id="status" class="status" style="display:none;"></div>
            <div id="progress" class="progress" style="display:none;"></div>
            <div id="adjustments"></div>
            <div id="history" class="history" style="display:none;"></div>
            <div id="output" class="output" style="display:none;"></div>
        </div>
        
        <script>
            let ws = null;
            let processingActive = false;
            
            function connectWebSocket() {
                ws = new WebSocket(`ws://${window.location.host}/ws`);
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
                
                ws.onclose = function() {
                    console.log('WebSocket closed');
                    setTimeout(connectWebSocket, 1000);
                };
            }
            
            function handleMessage(data) {
                const statusDiv = document.getElementById('status');
                const progressDiv = document.getElementById('progress');
                const adjustmentsDiv = document.getElementById('adjustments');
                const historyDiv = document.getElementById('history');
                const outputDiv = document.getElementById('output');
                
                switch(data.type) {
                    case 'start':
                        statusDiv.style.display = 'block';
                        statusDiv.innerHTML = `<strong>Status:</strong> ${data.message}`;
                        progressDiv.style.display = 'block';
                        processingActive = true;
                        document.getElementById('startBtn').disabled = true;
                        document.getElementById('adjustmentInput').disabled = false;
                        document.getElementById('adjustBtn').disabled = false;
                        break;
                    
                    case 'progress':
                        progressDiv.innerHTML = `<strong>Progress:</strong> ${data.message}`;
                        break;
                    
                    case 'adjustment':
                        const adjDiv = document.createElement('div');
                        adjDiv.className = 'adjustment';
                        adjDiv.innerHTML = `<strong>Adjustment #${data.count}:</strong> ${data.adjustment}<br>
                                           <strong>Updated Goal:</strong> ${data.combined_goal}`;
                        adjustmentsDiv.appendChild(adjDiv);
                        break;
                    
                    case 'cancellation':
                        adjustmentsDiv.innerHTML = '';
                        statusDiv.innerHTML = `<strong>Status:</strong> ${data.message}`;
                        break;
                    
                    case 'complete':
                        statusDiv.innerHTML = `<strong>Status:</strong> ${data.message}`;
                        progressDiv.style.display = 'none';
                        processingActive = false;
                        document.getElementById('startBtn').disabled = false;
                        document.getElementById('adjustmentInput').disabled = true;
                        document.getElementById('adjustBtn').disabled = true;
                        break;
                    
                    case 'output':
                        outputDiv.style.display = 'block';
                        outputDiv.innerHTML = `<strong>Final Output:</strong><br><br>${data.output}`;
                        break;
                    
                    case 'error':
                        statusDiv.innerHTML = `<strong>Error:</strong> ${data.message}`;
                        break;
                }
            }
            
            function startProcessing() {
                const input = document.getElementById('initialInput').value.trim();
                if (!input) {
                    alert('Please enter an initial task/query');
                    return;
                }
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'start', goal: input}));
                }
            }
            
            function sendAdjustment() {
                const input = document.getElementById('adjustmentInput').value.trim();
                if (!input) {
                    alert('Please enter an adjustment or new task');
                    return;
                }
                
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'adjustment', text: input}));
                    document.getElementById('adjustmentInput').value = '';
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('initialInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    startProcessing();
                }
            });
            
            document.getElementById('adjustmentInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendAdjustment();
                }
            });
            
            // Connect on page load
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "start":
                goal = message.get("goal")
                if agent.start_processing(goal):
                    await websocket.send_json({
                        "type": "start",
                        "message": f"Starting to work on: {goal}",
                        "goal": goal
                    })
                    
                    # Start sending progress updates
                    asyncio.create_task(send_progress_updates(websocket))
            
            elif message.get("type") == "adjustment":
                adjustment = message.get("text")
                if agent.send_adjustment(adjustment):
                    await websocket.send_json({
                        "type": "adjustment_received",
                        "message": f"Adjustment received: {adjustment}"
                    })
    
    except WebSocketDisconnect:
        agent.stop_processing()
        print("Client disconnected")


async def send_progress_updates(websocket: WebSocket):
    """Send progress updates from processing thread to UI."""
    while agent.processing_active or not agent.progress_updates.empty():
        try:
            update = agent.progress_updates.get(timeout=0.1)
            await websocket.send_json(update)
        except Empty:
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"Error sending update: {e}")
            break


if __name__ == "__main__":
    import uvicorn
    print("Starting web UI server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)

