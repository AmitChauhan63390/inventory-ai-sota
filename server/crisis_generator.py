import random
import copy

CRISIS_EVENTS = [
    {
        "id": "supplier_shutdown",
        "description": "Supplier {supplier_name} has announced a {duration}-day factory shutdown due to {reason}. They cannot fulfil any orders during this period.",
        "effect": "block_supplier",
        "duration_days": [7, 14],
        "variables": {
            "reason": ["factory fire", "labor strike", "regulatory inspection", "equipment failure"]
        }
    },
    {
        "id": "demand_spike",
        "description": "A viral social media trend has caused unexpected demand surge. Forecasting models predict {multiplier}x normal demand for the next {duration} days.",
        "effect": "multiply_demand",
        "duration_days": [3, 7],
        "variables": {
            "multiplier": [2, 3, 4]
        }
    },
    {
        "id": "expiring_stock",
        "description": "Quality control has flagged {quantity} units in current inventory as approaching expiry within 3 days. Unsold units will be written off at full cost.",
        "effect": "expiry_countdown",
        "duration_days": [3, 3],
        "variables": {
            "quantity": "random_fraction_of_inventory"
        }
    },
    {
        "id": "budget_freeze",
        "description": "Finance department has frozen procurement budget for {duration} days due to quarterly audit. No new orders can be placed.",
        "effect": "freeze_budget",
        "duration_days": [3, 5],
        "variables": {}
    },
    {
        "id": "logistics_delay",
        "description": "Regional flooding has disrupted logistics networks. All pending and new orders will experience additional {extra_days}-day delays regardless of supplier.",
        "effect": "add_global_delay",
        "duration_days": [4, 7],
        "variables": {
            "extra_days": [2, 3]
        }
    },
    {
        "id": "port_strike",
        "description": "A major port strike has halted all maritime traffic. All current pending orders are frozen for {duration} days, and new lead times are increased by 5 days.",
        "effect": "port_freeze",
        "duration_days": [5, 10],
        "variables": {}
    },
    {
        "id": "competitor_price_war",
        "description": "A major competitor has slashed prices. Market analysis suggests demand will drop by 70% unless you lower your price below {target_price} for the next {duration} days.",
        "effect": "price_war",
        "duration_days": [5, 5],
        "variables": {
            "target_price": [12, 14, 15]
        }
    }
]


EMERGENCY_ACTION_EFFECTS = {
    "none": "No emergency action. Standard operations continue.",
    "discount_expiring": "Discount expiring stock by 40%. Sells 2x faster but at 60% revenue.",
    "find_alternative": "Pay 30% premium to source from spot market. Ignores blocked suppliers.",
    "halt_orders": "Cancel all pending orders. Refund 80% of cost. Clears budget but empties pipeline."
}

class CrisisEventGenerator:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate_crisis(self) -> dict:
        template = self.rng.choice(CRISIS_EVENTS)
        event = copy.deepcopy(template)
        event["duration"] = self.rng.randint(event["duration_days"][0], event["duration_days"][1])
        
        # Resolve variables
        for k, v in event.get("variables", {}).items():
            if k == "reason" or k == "multiplier" or k == "extra_days":
                event[k] = self.rng.choice(v)
            if k == "quantity":
                event[k] = "half of all" # Will be replaced by actual logic in simulator if needed

        # Format description
        desc_kwargs = {"duration": event["duration"]}
        if "reason" in event: desc_kwargs["reason"] = event["reason"]
        if "multiplier" in event: desc_kwargs["multiplier"] = event["multiplier"]
        if "extra_days" in event: desc_kwargs["extra_days"] = event["extra_days"]
        if "quantity" in event: desc_kwargs["quantity"] = event["quantity"]
        desc_kwargs["supplier_name"] = "BudgetCo" # default for template

        event["active_description"] = event["description"].format(**desc_kwargs)
        return event

