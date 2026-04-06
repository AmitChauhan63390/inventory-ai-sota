import os
import google.generativeai as genai
import json

class SOTAReasoningGrader:
    """
    Advanced LLM-as-a-Judge grader.
    Uses a strong model (Gemini 1.5 Pro) to grade the agent's reasoning.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-pro")
        else:
            self.model = None

    def grade(self, reasoning: str, observation: dict, action: dict) -> float:
        """
        Grades the reasoning based on supply chain principles and the current state.
        Returns a score from 0.0 to 1.0.
        """
        if not self.model or not reasoning or len(reasoning) < 10:
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
            response = self.model.generate_content(prompt)
            # Simple extractor for JSON
            import re
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return float(data.get("score", 0.5))
        except Exception as e:
            print(f"SOTA Grading Error: {e}")
            
        return 0.5 # Fallback
