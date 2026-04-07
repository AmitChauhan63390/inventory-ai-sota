import asyncio
import os
import json
import re
import time
import httpx
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

# ---------------------------------------------------------
# MANDATORY CONFIGURATION (Meta OpenEnv Spec)
# ---------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
BENCHMARK = "InventoryAI-v1"
LOCAL_URL = os.getenv("LOCAL_URL") or "http://localhost:7860"

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
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------
# ENVIRONMENT CLIENT (Async/HTTP wrapper)
# ---------------------------------------------------------
class InventoryAIClient:
    """Async client matching the OpenEnv await step() style."""
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=60.0)

    async def reset(self, task_id: int):
        resp = await self._client.post(f"{LOCAL_URL}/reset", json={"task_id": task_id})
        return resp.json()

    async def step(self, action_dict: dict):
        resp = await self._client.post(f"{LOCAL_URL}/step", json=action_dict)
        return resp.json()
    
    async def close(self):
        await self._client.aclose()


# ---------------------------------------------------------
# AGENT LOGIC
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are a SOTA Inventory Management AI. Your goal is to maximize profit while maintaining 100% service level.
You must analyze inventory levels, batch perishability, and demand forecasts.

For Task 3 (Crisis): You must respond strategically to events like supplier shutdowns or demand spikes.

Respond ONLY with JSON:
{
    "order_quantity": int,
    "sales_price": float,
    "supplier_id": 1 | 2 | 3,
    "emergency_action": "none" | "discount_expiring" | "find_alternative" | "halt_orders",
    "reasoning": "short justification"
}
"""

async def get_model_action(client: AsyncOpenAI, obs: dict, retries=3):
    """Call the LLM with retry logic and fallback."""
    for attempt in range(retries):
        try:
            completion = await client.chat.completions.create(
                model=MODEL_NAME,

                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(obs)}
                ],
                response_format={ "type": "json_object" },
                temperature=0.4,
                timeout=25.0
            )
            content = completion.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            if attempt < retries - 1:
                print(f"[DEBUG] Model call failed (attempt {attempt+1}/{retries}): {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)
                continue
            # Safe fallback logic
            return {"order_quantity": 25, "sales_price": 20.0, "supplier_id": 2, "emergency_action": "none", "reasoning": f"fallback due to error: {e}"}

async def run_task(task_id: int):
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env = InventoryAIClient()
    
    task_name = f"task_{task_id}"
    log_start(task=task_name, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    info = {}
    
    try:
        obs = await env.reset(task_id=task_id)
        done = False
        
        for step_idx in range(1, 51): # 50 max days
            try:
                action_data = await get_model_action(client, obs)

                
                result = await env.step(action_data)
                if "observation" not in result:
                    print(f"[DEBUG] Malformed response in step {step_idx}: {result}")
                    break
                    
                obs = result["observation"]
                reward = result["reward"]
                done = result["done"]
                info = result.get("info", {})
                
                rewards.append(reward)
                steps_taken = step_idx
                
                # Action string for logs should be compact
                action_str = f"order={action_data.get('order_quantity')} price={action_data.get('sales_price')}"
                log_step(step=step_idx, action=action_str, reward=reward, done=done, error=None)
                
                if done: break
            except Exception as loop_e:
                print(f"[DEBUG] Error in step {step_idx}: {loop_e}")
                continue
            
        # Terminal stats from info
        res = info.get("episode_result", {})
        score = res.get("final_score", 0.0) 
        success = score >= 0.1 
        
    except Exception as e:
        print(f"[DEBUG] Critical Crash during task: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        await env.close()

async def main():
    # Sequence through all 3 mandatory tasks
    for tid in [1, 2, 3]:
        await run_task(tid)
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
