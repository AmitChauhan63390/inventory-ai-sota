import random

SUPPLIERS = {
    1: {
        "name": "BudgetCo",
        "cost_per_unit": 8.0,
        "lead_time_days": 5,
        "reliability": 0.75,
    },
    2: {
        "name": "ReliableMart",
        "cost_per_unit": 12.0,
        "lead_time_days": 3,
        "reliability": 0.97,
    },
    3: {
        "name": "ExpressLogix",
        "cost_per_unit": 18.0,
        "lead_time_days": 1,
        "reliability": 1.0,
    }
}

class SupplierRegistry:
    def get_suppliers(self, task_id: int):
        if task_id == 1:
            return {} # Task 1 doesn't expose suppliers, hardcodes to Supplier 2
        return SUPPLIERS
        
    def resolve_lead_time_and_delay(self, supplier_id: int, rng: random.Random) -> tuple[int, bool]:
        supplier = SUPPLIERS.get(supplier_id, SUPPLIERS[2])
        lead_time = supplier["lead_time_days"]
        delayed = False
        if rng.random() > supplier["reliability"]:
            delay = rng.randint(1, 2)
            lead_time += delay
            delayed = True
        return lead_time, delayed
