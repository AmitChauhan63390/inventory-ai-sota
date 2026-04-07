import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from models import WarehouseAction, WarehouseObservation
from server.simulator import InventorySimulator
import json

app = FastAPI(title="InventoryAI-v1 Benchmark", version="2.1.0")

# Storage for environments by session or generic 
# Simplification for single-user FastAPI setup
current_env = None

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>InventoryAI-v1 Benchmark</title>
            <style>
                body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                .card { background: #1e293b; padding: 2rem; border-radius: 1rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); border: 1px solid #334155; max-width: 500px; text-align: center; }
                h1 { color: #38bdf8; margin-top: 0; }
                .status { display: inline-block; padding: 0.25rem 0.75rem; background: #064e3b; color: #4ade80; border-radius: 9999px; font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem; }
                p { line-height: 1.6; color: #94a3b8; }
                .btn { display: inline-block; margin-top: 1.5rem; padding: 0.75rem 1.5rem; background: #38bdf8; color: #0f172a; border-radius: 0.5rem; text-decoration: none; font-weight: 600; transition: background 0.2s; }
                .btn:hover { background: #7dd3fc; }
                .footer { margin-top: 2rem; font-size: 0.75rem; color: #64748b; }
            </style>
        </head>
        <body>
            <div class="card">
                <div class="status">● System Operational</div>
                <h1>InventoryAI-v1</h1>
                <p>A High-Fidelity SOTA Supply Chain Benchmark for Meta OpenEnv.</p>
                <div style="text-align: left; background: #0f172a; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                    <code style="color: #38bdf8;">POST /reset</code><br>
                    <code style="color: #38bdf8;">POST /step</code>
                </div>
                <a href="/docs" class="btn">View API Documentation</a>
                <div class="footer">Developed by Amit and Rishabh</div>
            </div>
        </body>
    </html>
    """

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: Optional[int] = None

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None, task_id: Optional[int] = Query(None)):
    global current_env
    
    # Priority: 
    # 1. Query parameter
    # 2. Body parameter
    # 3. Default (1)
    
    final_task_id = 1
    final_seed = None
    
    if task_id is not None:
        final_task_id = task_id
    elif req is not None:
        final_task_id = req.task_id
        final_seed = req.seed
        
    current_env = InventorySimulator(task_id=int(final_task_id), seed=final_seed)
    obs = current_env.reset()
    return obs.model_dump()

@app.get("/state")
def state():
    global current_env
    if not current_env:
        current_env = InventorySimulator(task_id=1)
    obs = current_env._get_observation()
    return obs.model_dump()

@app.post("/step")
def step(action: dict): # accept dictionary to be backward compatible and dynamic
    global current_env
    if not current_env:
        current_env = InventorySimulator(task_id=1)
        
    try:
        # Fallback to defaults
        if "order_quantity" not in action:
            raise ValueError("order_quantity is required")
            
        # Ensure the action is one of the valid literals, otherwise default to "none"
        e_action = action.get("emergency_action", "none")
        final_e_action = str(e_action).lower()
        if final_e_action not in ["none", "discount_expiring", "find_alternative", "halt_orders"]:
            final_e_action = "none"

        validated_action = WarehouseAction(
            order_quantity=int(action.get("order_quantity", 0)),
            sales_price=float(action.get("sales_price", 20.0)),
            supplier_id=int(action.get("supplier_id", 1 if current_env.task_id == 1 else 2)),
            emergency_action=final_e_action,
            reasoning=str(action.get("reasoning", ""))
        )

    except Exception as e:
        # Handle simple integer passing (backward compatibility)
        if isinstance(action, int):
            validated_action = WarehouseAction(order_quantity=action)
        else:
            raise e

    obs, reward, done, info = current_env.step(validated_action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": done,
        "info": info
    }

@app.get("/health")
def health():
    return {"status": "ok"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
