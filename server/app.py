import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from models import WarehouseAction, WarehouseObservation
from server.simulator import InventorySimulator
import json

app = FastAPI(title="Inventory AI Environment - Upgraded")

# Storage for environments by session or generic 
# Simplification for single-user FastAPI setup
current_env = None

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: Optional[int] = None

@app.post("/reset")
def reset(req: ResetRequest):
    global current_env
    current_env = InventorySimulator(task_id=req.task_id, seed=req.seed)
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
            
        validated_action = WarehouseAction(
            order_quantity=int(action.get("order_quantity", 0)),
            sales_price=float(action.get("sales_price", 20.0)), # Default to market price
            supplier_id=int(action.get("supplier_id", 1 if current_env.task_id == 1 else 2)),
            emergency_action=action.get("emergency_action", "none"),
            reasoning=action.get("reasoning", "")
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
