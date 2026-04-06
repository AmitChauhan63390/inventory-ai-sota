import numpy as np
import random
from typing import Tuple, Dict, Any, List

from models import WarehouseObservation, WarehouseAction, Reward, RewardMetrics, WarehouseState, EpisodeResult, Batch
from .suppliers import SupplierRegistry, SUPPLIERS
from .crisis_generator import CrisisEventGenerator, EMERGENCY_ACTION_EFFECTS
from .graders import grade_task_1, grade_task_2, grade_task_3, grade_crisis_response
from .reasoning_grader import ReasoningGrader
from .reasoning_grader_sota import SOTAReasoningGrader
from .bullwhip import calculate_bullwhip_coefficient

class InventorySimulator:
    def __init__(self, task_id: int = 1, seed: int = None):
        self.task_id = task_id
        if seed is not None:
            self.rng = random.Random(seed)
            np.random.seed(seed)
        else:
            self.rng = random.Random()
            
        self.config = {
            "max_days": 50,
            "base_market_price": 20.0,
            "unit_cost": 10.0,
            "holding_cost_rate": 0.5,
            "overflow_cost_rate": 2.0, # Cost per unit over capacity
            "capacity": 300,
            "stockout_penalty": 5.0,
            "base_demand": 20.0,
            "seasonal_amplitude": 10.0,
            "seasonal_period": 30,
            "demand_std": 5.0 if task_id == 1 else 10.0,
            "price_elasticity": 1.5, # Demand drops faster as price increases
            "default_expiry_days": 15,
        }
        
        if task_id == 1:
            self.task_description = "Safety First (Easy). Single product, Supplier 2 only. Graded purely on Customer Service Level."
        elif task_id == 2:
            self.task_description = "Multi-Supplier Profit Hero (Medium). All 3 suppliers available. Budget constraint active. Graded on profit and supplier efficiency."
        else:
            self.task_description = "SOTA Crisis Manager (Hard). Complex supply chain with perishability, price elasticity, and market intelligence."
            
        self.supplier_registry = SupplierRegistry()
        self.crisis_generator = CrisisEventGenerator(self.rng)
        self.reasoning_grader = ReasoningGrader()
        self.sota_grader = SOTAReasoningGrader()
        self.reset()
        
    def _generate_base_demand(self, day: int) -> float:
        """Baseline demand without pricing effects."""
        seasonal = self.config["seasonal_amplitude"] * np.sin(2 * np.pi * day / self.config["seasonal_period"])
        
        multiplier = 1.0
        if self.state.active_crisis_id == "demand_spike":
            multiplier = self.active_crisis_event.get("multiplier", 2)
            
        return max(0.0, float((self.config["base_demand"] + seasonal) * multiplier))

    def _apply_price_elasticity(self, base_demand: float, set_price: float) -> float:
        """Adjust demand based on the agent's set price relative to market price."""
        market_price = self.config["base_market_price"]
        # Formula: D = D0 * (P_m / P_s) ^ elasticity
        ratio = market_price / max(1.0, set_price)
        elasticity = self.config["price_elasticity"]
        return base_demand * (ratio ** elasticity)

    def _get_actual_demand(self, forecast: float) -> float:
        return max(0.0, float(forecast + np.random.normal(0, self.config["demand_std"])))

    def reset(self) -> WarehouseObservation:
        # Starting with 50 units in 2 batches
        initial_batches = [
            Batch(quantity=25, expiry_day=10, arrival_day=-5),
            Batch(quantity=25, expiry_day=15, arrival_day=-2)
        ]
        
        self.state = WarehouseState(
            inventory_batches=initial_batches,
            pending_orders=[],
            procurement_budget=5000.0 if self.task_id in [2, 3] else 999999.0,
            total_revenue=0.0,
            total_cost=0.0,
            stockout_days=0,
            current_day=0,
            active_crisis=None,
            crisis_day_remaining=None,
            active_crisis_id=None,
            delayed_orders=[],
            bullwhip_history=[],
            demand_history=[],
            task_id=self.task_id,
            intelligence_history=[]
        )
        
        # Initial Forecast
        base_demand = self._generate_base_demand(0)
        self.demand_today = self._get_actual_demand(base_demand) # simplified for day 0
        
        self.active_crisis_event = None
        self.total_orders = 0
        self.total_delays = 0
        self.total_crisis_score = 0.0
        self.total_reasoning_score = 0.0
        self.crisis_trigger_day = self.rng.randint(10, 40) if self.task_id == 3 else -1
        
        return self._get_observation()

    def _get_current_inventory_total(self) -> int:
        return sum(b.quantity for b in self.state.inventory_batches)

    def _get_observation(self) -> WarehouseObservation:
        available_suppliers = self.supplier_registry.get_suppliers(self.task_id)
        current_total = self._get_current_inventory_total()
        
        service_level = 1.0 if self.state.current_day == 0 else max(0.0, 1.0 - (self.state.stockout_days / self.state.current_day))
        bullwhip = calculate_bullwhip_coefficient(self.state.bullwhip_history, self.state.demand_history)
        
        # Intelligence Feed logic
        feed = []
        if self.task_id == 3:
            # Show the last 2 items from intelligence history
            feed = self.state.intelligence_history[-2:]
            
            # Foreshadowing logic: if crisis is 3 days away, add hint
            if self.crisis_trigger_day != -1:
                days_until = self.crisis_trigger_day - self.state.current_day
                if 1 <= days_until <= 3:
                    feed.append(f"MARKET INTEL: Logistics analysts warn of potential disruptions starting in {days_until} days.")
        
        return WarehouseObservation(
            current_inventory=current_total,
            inventory_batches=self.state.inventory_batches,
            warehouse_capacity=self.config["capacity"],
            current_day=self.state.current_day,
            simulation_length=self.config["max_days"],
            demand_today=int(self.demand_today),
            demand_forecast_tomorrow=int(self._generate_base_demand(self.state.current_day + 1)),
            demand_forecast_3day=int(self._generate_base_demand(self.state.current_day + 3)),
            current_market_price=self.config["base_market_price"],
            pending_orders=[{"quantity": o["quantity"], "arrives_on_day": self.state.current_day + o["days_remaining"], "supplier_id": o["supplier_id"], "delayed": o.get("delayed", False)} for o in self.state.pending_orders],
            delayed_orders=[{"quantity": o["quantity"], "arrives_on_day": self.state.current_day + o["days_remaining"], "supplier_id": o["supplier_id"], "delayed": o.get("delayed", False)} for o in self.state.delayed_orders],
            available_suppliers=available_suppliers,
            procurement_budget=self.state.procurement_budget,
            total_revenue=self.state.total_revenue,
            total_cost=self.state.total_cost,
            net_profit=self.state.total_revenue - self.state.total_cost,
            service_level_so_far=max(0.0, service_level),
            stockout_days=self.state.stockout_days,
            bullwhip_coefficient=bullwhip,
            active_crisis=self.state.active_crisis,
            crisis_day_remaining=self.state.crisis_day_remaining,
            intelligence_feed=feed,
            task_id=self.task_id,
            task_description=self.task_description
        )

    def state(self) -> WarehouseState:
        """Required by OpenEnv spec: returns the full internal state."""
        return self.state

    def step(self, action: WarehouseAction) -> Tuple[WarehouseObservation, float, bool, dict]:
        # 1. Reasoning Grade
        if self.task_id == 3:
            reasoning_score = self.sota_grader.grade(action.reasoning, self._get_observation().model_dump(), action.model_dump())
        else:
            reasoning_score = self.reasoning_grader.score(action.reasoning, self.state.model_dump())
            
        self.total_reasoning_score += reasoning_score
        
        # 2. Crisis Progression
        if self.task_id == 3:
            if self.state.current_day == self.crisis_trigger_day:
                self.active_crisis_event = self.crisis_generator.generate_crisis()
                self.state.active_crisis_id = self.active_crisis_event["id"]
                self.state.active_crisis = self.active_crisis_event["active_description"]
                self.state.crisis_day_remaining = self.active_crisis_event["duration"]
                self.state.intelligence_history.append(f"URGENT: {self.state.active_crisis_id.replace('_', ' ').upper()} started.")
                
            if self.state.crisis_day_remaining is not None:
                self.state.crisis_day_remaining -= 1
                if self.state.crisis_day_remaining <= 0:
                    self.state.intelligence_history.append(f"UPDATE: {self.state.active_crisis_id.replace('_', ' ').upper()} has been resolved.")
                    self.state.active_crisis = None
                    self.state.active_crisis_id = None
                    self.state.crisis_day_remaining = None
                    self.active_crisis_event = None

        # 3. Process Actions (Pricing & Ordering)
        qty = int(action.order_quantity)
        set_price = float(action.sales_price)
        supplier_id = action.supplier_id if self.task_id in [2, 3] else 2
        
        # Emergency Effects
        is_order_allowed = True
        cost_multiplier = 1.0
        
        if self.state.active_crisis_id == "budget_freeze":
            is_order_allowed = False
        
        if action.emergency_action == "find_alternative" and self.task_id == 3:
            cost_multiplier = 1.3
            is_order_allowed = True
            
        if action.emergency_action == "halt_orders" and self.task_id == 3:
            qty = 0
            for order in self.state.pending_orders:
                refund = order["ordered_cost"] * 0.8
                self.state.procurement_budget += refund
                self.state.total_cost -= refund
            self.state.pending_orders = []

        # 4. Supplier Interaction
        if qty > 0 and is_order_allowed:
            supplier = SUPPLIERS.get(supplier_id, SUPPLIERS[2])
            cost = qty * supplier["cost_per_unit"] * cost_multiplier
            
            if cost <= self.state.procurement_budget:
                self.state.procurement_budget -= cost
                lead_time, delayed = self.supplier_registry.resolve_lead_time_and_delay(supplier_id, self.rng)
                
                if self.state.active_crisis_id == "logistics_delay":
                    lead_time += self.active_crisis_event.get("extra_days", 2)
                    
                order_item = {
                    "quantity": qty, 
                    "days_remaining": lead_time, 
                    "supplier_id": supplier_id, 
                    "delayed": delayed, 
                    "ordered_cost": cost,
                    "expiry_days": self.config["default_expiry_days"]
                }
                self.state.pending_orders.append(order_item)
                if delayed:
                    self.state.delayed_orders.append(order_item)
                    self.total_delays += 1
                self.total_orders += 1
            else:
                qty = 0 # Budget limit hit
                
        self.state.bullwhip_history.append(qty)

        # 5. Advance Time & Delivery
        arrived_stock_batches = []
        for order in self.state.pending_orders:
            order["days_remaining"] -= 1
            if order["days_remaining"] <= 0:
                arrived_stock_batches.append(Batch(
                    quantity=order["quantity"],
                    expiry_day=self.state.current_day + order["expiry_days"],
                    arrival_day=self.state.current_day
                ))
                if order in self.state.delayed_orders:
                    self.state.delayed_orders.remove(order)
                    
        self.state.pending_orders = [o for o in self.state.pending_orders if o["days_remaining"] > 0]
        self.state.inventory_batches.extend(arrived_stock_batches)

        # 6. Expiry Logic (Before Sales)
        expiry_loss = 0.0
        valid_batches = []
        for batch in self.state.inventory_batches:
            if batch.expiry_day <= self.state.current_day:
                expiry_loss += batch.quantity * self.config["unit_cost"]
            else:
                valid_batches.append(batch)
        self.state.inventory_batches = valid_batches

        # 7. Sales Logic (FIFO)
        # Demand is affected by current day's price set by action
        base_demand = self._generate_base_demand(self.state.current_day)
        realized_demand = self._apply_price_elasticity(base_demand, set_price)
        self.demand_today = int(realized_demand) # used for history and observation
        
        total_available = self._get_current_inventory_total()
        sales = min(float(total_available), realized_demand)
        unmet_demand = max(0.0, realized_demand - sales)
        
        if unmet_demand > 0:
            self.state.stockout_days += 1
            
        # Deduct from batches using FIFO
        remaining_to_sell = int(sales)
        updated_batches = []
        # Sort by expiry day to ensure FIFO/FEFO
        self.state.inventory_batches.sort(key=lambda x: x.expiry_day)
        
        for batch in self.state.inventory_batches:
            if remaining_to_sell > 0:
                if batch.quantity <= remaining_to_sell:
                    remaining_to_sell -= batch.quantity
                    # Batch is fully sold
                else:
                    batch.quantity -= remaining_to_sell
                    remaining_to_sell = 0
                    updated_batches.append(batch)
            else:
                updated_batches.append(batch)
        self.state.inventory_batches = updated_batches
        self.state.demand_history.append(int(realized_demand))

        # 8. Financials & Rewards
        revenue = sales * set_price
        purchase_cost = qty * (SUPPLIERS.get(supplier_id, SUPPLIERS[2])["cost_per_unit"]) * cost_multiplier if qty > 0 else 0.0
        
        current_inv = self._get_current_inventory_total()
        holding_cost = current_inv * self.config["holding_cost_rate"]
        
        # Overflow Cost
        overflow_units = max(0, current_inv - self.config["capacity"])
        overflow_cost = overflow_units * self.config["overflow_cost_rate"]
        
        stockout_penalty = unmet_demand * self.config["stockout_penalty"]
        shaping = 5.0 if 10 <= current_inv <= 50 else (-10.0 if current_inv == 0 else 0)
        
        # Bullwhip Effect Penalty (Novelty requirement): penalize erratic ordering
        # If orders are jumping between 0 and 200, it's bad for the supply chain.
        bullwhip_penalty = 0.0
        if len(self.state.bullwhip_history) > 1:
            order_variance = abs(action.order_quantity - self.state.bullwhip_history[-2])
            if order_variance > 100:
                bullwhip_penalty = 20.0 # Significant penalty for erratic orders
        
        step_reward = float(revenue - purchase_cost - holding_cost - overflow_cost - stockout_penalty - expiry_loss + shaping - bullwhip_penalty)
        
        self.state.total_revenue += revenue
        self.state.total_cost += (purchase_cost + holding_cost + overflow_cost + stockout_penalty + expiry_loss + bullwhip_penalty)
        
        # 9. Completion Check
        self.state.current_day += 1
        done = self.state.current_day >= self.config["max_days"]
        
        metrics = RewardMetrics(
            revenue=revenue,
            purchase_cost=purchase_cost,
            holding_cost=holding_cost,
            stockout_penalty=stockout_penalty,
            overflow_cost=overflow_cost,
            expiry_loss=expiry_loss,
            shaping=shaping
        )
        
        if done:
            net_profit = self.state.total_revenue - self.state.total_cost
            service_level = max(0.0, 1.0 - (self.state.stockout_days / max(1, self.config["max_days"])))
            avg_reasoning = self.total_reasoning_score / max(1, self.config["max_days"])
            bullwhip = calculate_bullwhip_coefficient(self.state.bullwhip_history, self.state.demand_history)
            
            episode_result = EpisodeResult(
                net_profit=net_profit,
                service_level=service_level,
                total_delays=self.total_delays,
                total_orders=max(1, self.total_orders),
                avg_crisis_response_score=0.5,
                bullwhip_coefficient=bullwhip,
                avg_reasoning_score=avg_reasoning,
                final_score=0.0
            )

            if self.task_id == 1:
                episode_result.final_score = grade_task_1(episode_result)
            elif self.task_id == 2:
                episode_result.final_score = grade_task_2(episode_result)
            else:
                episode_result.final_score = grade_task_3(episode_result)
                
            info = {
                "episode_result": episode_result.model_dump(),
                "metrics": metrics.model_dump()
            }
        else:
            info = {"metrics": metrics.model_dump()}
            
        return self._get_observation(), step_reward, done, info

