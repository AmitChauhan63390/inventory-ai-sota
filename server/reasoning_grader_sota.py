import os
import json
from openai import OpenAI

class SOTAReasoningGrader:
    """
    Advanced LLM-as-a-Judge grader.
    Uses the OpenAI Client (standardized for OpenEnv hackathon compliance).
    Grades the agent's reasoning based on supply chain principles.
    """
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self.base_url = base_url or os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
        self.model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
        
        if self.api_key:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = None

    def grade(self, reasoning: str, observation: dict, action: dict) -> float:
        """
        Grades the reasoning based on supply chain principles and the current state.
        Returns a score from 0.0 to 1.0.
        """
        if not self.client or not reasoning or len(reasoning) < 10:
            return 0.5 # Default middle score if no model or reasoning
        
        prompt = f"""
        You are a Supply Chain Professor grading a student's daily warehouse decision.
        
        CONTEXT (Observation):
        {json.dumps(observation, indent=2)}
        
        STUDENT ACTION:
        {json.dumps(action, indent=2)}
        
        STUDENT REASONING:
        "{reasoning}"
        
        GRADE the reasoning on a scale of 0.0 to 1.0 based on these criteria:
        1. STRATEGIC FORESIGHT: Did they account for lead times and pending arrivals?
        2. ECONOMIC LOGIC: Did they balance holding costs vs stockout penalties?
        3. CRISIS HANDLING: If a crisis is active or hinted in intel, did they react correctly?
        4. PRICING STRATEGY: Did they adjust price based on inventory levels (e.g. lowering price for expiring stock)?
        
        Output ONLY a JSON object: {{"score": float, "critique": "short explanation"}}
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                temperature=0.0
            )
            data = json.loads(completion.choices[0].message.content)
            return float(data.get("score", 0.5))
        except Exception as e:
            # Fallback for models that don't support response_format="json_object"
            try:
                # Basic string processing just in case
                import re
                text = completion.choices[0].message.content
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return float(data.get("score", 0.5))
            except:
                pass
            print(f"SOTA Grading Error: {e}")
            
        return 0.5 # Fallback

