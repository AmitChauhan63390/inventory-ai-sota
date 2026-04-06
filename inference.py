import asyncio
import os
import json
import re
import time
import httpx
from typing import List, Optional, Dict, Any
from openai import OpenAI

# ---------------------------------------------------------
# MANDATORY CONFIGURATION (Meta OpenEnv Spec)
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
BENCHMARK = "InventoryAI-v1"
LOCAL_URL = "http://localhost:7860" # Internal container port

# ---------------------------------------------------------
# LOGGING HELPERS (Strict format enforcement)
# ---------------------------------------------------------
def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # reward list joined by commas, formatted to 2 decimals
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------
# ENVIRONMENT CLIENT (Async/HTTP wrapper)
# ---------------------------------------------------------
class InventoryAIClient:
    """Async client matching the OpenEnv await step() style."""
    async def reset(self, task_id: int):
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{LOCAL_URL}/reset", json={"task_id": task_id}, timeout=30.0)
            return resp.json()

    async def step(self, action_dict: dict):
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{LOCAL_URL}/step", json=action_dict, timeout=30.0)
            return resp.json()

# ---------------------------------------------------------
# AGENT LOGIC
# ---------------------------------------------------------
def get_model_action(client: OpenAI, obs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
    TASK: {obs.get('task_description')} (Day {obs.get('current_day')})
    INV: {obs.get('current_inventory')} / Capacity: {obs.get('warehouse_capacity')}
    BATCHES (Exp Tracking): {json.dumps(obs.get('inventory_batches', []))}
    FORECAST: Tomorrow: {obs.get('demand_forecast_tomorrow')}, 3-Day: {obs.get('demand_forecast_3day')}
    INTEL FEED: {obs.get('intelligence_feed')}
    ACTIVE CRISIS: {obs.get('active_crisis')}
    
    GOAL: Maximize Profit & Service Level.
    Respond ONLY with JSON:
    {{
        "order_quantity": int,
        "sales_price": float (5.0-50.0),
        "supplier_id": 1-3,
        "emergency_action": "none" | "discount_expiring" | "find_alternative" | "halt_orders",
        "reasoning": "short justification"
    }}
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.2,
            max_tokens=200,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        # Safe fallback logic
        return {"order_quantity": 25, "sales_price": 20.0, "supplier_id": 2, "emergency_action": "none", "reasoning": "fallback"}

async def run_task(task_id: int):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = InventoryAIClient()
    
    task_name = f"task_{task_id}"
    log_start(task=task_name, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    
    try:
        obs = await env.reset(task_id=task_id)
        done = False
        
        for step_idx in range(1, 51): # 50 max days
            action_data = get_model_action(client, obs)
            
            result = await env.step(action_data)
            obs = result["observation"]
            reward = result["reward"]
            done = result["done"]
            info = result["info"]
            
            rewards.append(reward)
            steps_taken = step_idx
            
            # Action string for logs should be compact
            action_str = f"order={action_data.get('order_quantity')} price={action_data.get('sales_price')}"
            log_step(step=step_idx, action=action_str, reward=reward, done=done, error=None)
            
            if done: break
            
        # Terminal stats from info
        score = info.get("grade", 0.0) # Our simulator returns grade in info
        success = score >= 0.1 # Per sample Success threshold
        
    except Exception as e:
        # Log error in step if it occurred during loop
        print(f"[DEBUG] Crash during task: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    # Sequence through all 3 mandatory tasks
    for tid in [1, 2, 3]:
        await run_task(tid)
        await asyncio.sleep(2) # Throttle

if __name__ == "__main__":
    asyncio.run(main())
