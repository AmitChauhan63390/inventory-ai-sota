from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Dict, Literal, Optional, Any

class Batch(BaseModel):
    quantity: int = Field(..., description="Number of units in this batch")
    expiry_day: int = Field(..., description="Day this batch expires and becomes unsellable")
    arrival_day: int = Field(..., description="Day this batch arrived in warehouse")

class Action(BaseModel):
    pass

class Observation(BaseModel):
    pass

class State(BaseModel):
    pass

class WarehouseAction(Action):
    order_quantity: int = Field(
        ge=0, le=200,
        description="Units to order this day (0-200)"
    )
    sales_price: float = Field(
        default=20.0,
        ge=5.0, le=50.0,
        description="Set the sales price for today. Higher prices reduce demand volume."
    )
    supplier_id: int = Field(
        default=2,
        ge=1, le=3,
        description="Which supplier to order from. 1=Cheap&Slow, 2=Balanced, 3=Fast&Expensive"
    )
    emergency_action: Literal["none", "discount_expiring", "find_alternative", "halt_orders"] = Field(
        default="none",
        description="Emergency action to take this tick if a crisis event is active"
    )
    reasoning: str = Field(
        default="",
        max_length=500,
        description="Agent's brief explanation of why it made this decision. Used for partial grading."
    )

    @field_validator("order_quantity")
    def quantity_must_be_valid(cls, v):
        if not isinstance(v, int):
            raise ValueError("order_quantity must be an integer")
        return v

class WarehouseObservation(Observation):
    """
    Detailed observation state returned each day in the simulation.
    Agents use this information to decide order quantities, pricing, and emergency actions.
    """
    # Core state
    current_inventory: int = Field(..., description="Total count of available units across all batches")
    inventory_batches: List[Batch] = Field(..., description="List of specific stock batches and their expiry days")
    warehouse_capacity: int = Field(default=300, description="Maximum units the warehouse can store without overflow fees")
    current_day: int = Field(..., description="Current simulation day (0-50)")
    simulation_length: int = Field(..., description="Total length of the simulation")

    # Demand information
    demand_today: int = Field(..., description="Actual client demand that materialized today (after pricing)")
    demand_forecast_tomorrow: int = Field(..., description="Predicted demand for the next day at 'base price'")
    demand_forecast_3day: int = Field(..., description="Predicted demand 3 days from now at 'base price'")
    current_market_price: float = Field(default=20.0, description="Baseline market price for demand comparison")

    # Order pipeline
    pending_orders: list[dict] = Field(..., description="Orders waiting to arrive")
    delayed_orders: list[dict] = Field(..., description="Orders flagged as experiencing delays")

    # Supplier information
    available_suppliers: dict = Field(..., description="Details of available suppliers")
    procurement_budget: float = Field(..., description="Remaining cash for purchasing stock")

    # Financial
    total_revenue: float = Field(..., description="Total money earned from sales")
    total_cost: float = Field(..., description="Total money spent on goods, penalties, and storage")
    net_profit: float = Field(..., description="Total revenue - Total cost")

    # Performance metrics
    service_level_so_far: float = Field(..., description="Percentage of days fully meeting demand")
    stockout_days: int = Field(..., description="Number of days demand exceeded stock")
    bullwhip_coefficient: float = Field(..., description="Measure of order volatility vs demand volatility")

    # Crisis & Intel
    active_crisis: Optional[str] = Field(None, description="Plain text explanation of active crisis event")
    crisis_day_remaining: Optional[int] = Field(None, description="Days remaining before crisis ends")
    intelligence_feed: List[str] = Field(default_factory=list, description="Market rumors and hints about upcoming logistics or demand shifts")

    # Task context
    task_id: int = Field(..., description="Current simulation task ID")
    task_description: str = Field(..., description="Brief description of the current task goal")

class WarehouseState(State):
    """
    Internal simulator tracking state. Hidden from the agent.
    """
    inventory_batches: List[Batch]
    pending_orders: list[dict]
    procurement_budget: float
    total_revenue: float
    total_cost: float
    stockout_days: int
    current_day: int
    active_crisis: Optional[str]
    crisis_day_remaining: Optional[int]
    active_crisis_id: Optional[str]
    delayed_orders: list[dict]
    bullwhip_history: list[int]
    demand_history: list[int]
    task_id: int
    intelligence_history: List[str]
    
class EpisodeResult(BaseModel):
    net_profit: float
    service_level: float
    total_delays: int
    total_orders: int
    avg_crisis_response_score: float
    bullwhip_coefficient: float
    avg_reasoning_score: float
    final_score: float

class RewardMetrics(BaseModel):
    revenue: float
    purchase_cost: float
    holding_cost: float
    stockout_penalty: float
    overflow_cost: float
    expiry_loss: float
    shaping: float

class Reward(BaseModel):
    value: float = Field(..., description="The scalar reward signal for the step")
    is_partial: bool = Field(True, description="Indicates if this is a step reward")
    metrics: RewardMetrics = Field(..., description="Detailed breakdown of the reward components")

