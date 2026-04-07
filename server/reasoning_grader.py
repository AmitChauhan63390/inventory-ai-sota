import re

class ReasoningGrader:
    """
    Scores agent reasoning quality deterministically.
    Does NOT use an LLM. Pure rule-based.
    """

    INVENTORY_KEYWORDS = ["stock", "inventory", "units", "warehouse"]
    DEMAND_KEYWORDS = ["demand", "forecast", "customer", "sales"]
    SUPPLIER_KEYWORDS = ["supplier", "lead time", "delivery", "order"]
    FINANCIAL_KEYWORDS = ["cost", "budget", "profit", "revenue", "price"]
    CRISIS_KEYWORDS = ["crisis", "emergency", "shortage", "spike", "freeze", "expir", "delay"]
    QUANTITATIVE_PATTERN = r'\d+'  # did agent mention actual numbers?

    def score(self, reasoning: str, context: dict) -> float:
        """
        Returns 0.0 to 1.0 based on:
        - Did agent reference relevant domain concepts?
        - Did agent mention actual numbers from the observation?
        - Did agent acknowledge active crisis if one exists?
        - Is reasoning non-empty and substantial (>20 chars)?
        """
        if not reasoning or len(reasoning) < 20:
            return 0.0001

        score = 0.0
        reasoning_lower = reasoning.lower()

        # Concept coverage (0.4 total)
        concept_groups = [
            self.INVENTORY_KEYWORDS,
            self.DEMAND_KEYWORDS,
            self.SUPPLIER_KEYWORDS,
            self.FINANCIAL_KEYWORDS,
        ]
        concepts_covered = sum(
            any(kw in reasoning_lower for kw in group)
            for group in concept_groups
        )
        score += (concepts_covered / len(concept_groups)) * 0.4

        # Quantitative reasoning (0.2)
        numbers_mentioned = re.findall(self.QUANTITATIVE_PATTERN, reasoning)
        if len(numbers_mentioned) >= 2:
            score += 0.2
        elif len(numbers_mentioned) == 1:
            score += 0.1

        # Crisis acknowledgment (0.3) — only if crisis is active
        if context.get("active_crisis"):
            crisis_acknowledged = any(
                kw in reasoning_lower for kw in self.CRISIS_KEYWORDS
            )
            score += 0.3 if crisis_acknowledged else 0.0
        else:
            score += 0.3  # full marks if no crisis to acknowledge

        # Length/substance bonus (0.1)
        if len(reasoning) >= 100:
            score += 0.1
        elif len(reasoning) >= 50:
            score += 0.05

        return round(max(0.0001, min(0.9999, score)), 4)
