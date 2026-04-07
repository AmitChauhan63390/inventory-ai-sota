def normalize(val: float, min_val: float, max_val: float) -> float:
    return max(0.0, min(1.0, (val - min_val) / (max_val - min_val)))

def strict_clamp(val: float) -> float:
    """Ensure the score is strictly between 0 and 1 (exclusive)."""
    return max(0.0001, min(0.9999, float(val)))

def grade_task_1(episode_result) -> float:
    return strict_clamp(round(episode_result.service_level, 4))

def grade_task_2(episode_result) -> float:
    profit_score = normalize(episode_result.net_profit, min_val=-2000, max_val=3000)
    service_score = episode_result.service_level  # 0.0 to 1.0
    supplier_efficiency = 1.0 - (episode_result.total_delays / max(episode_result.total_orders, 1))

    final_score = (
        0.50 * profit_score +
        0.30 * service_score +
        0.20 * supplier_efficiency
    ) * (0.90) + (0.10 * episode_result.avg_reasoning_score) # As mentioned "Contributes 10% to Task 2 final score"
    return strict_clamp(round(final_score, 4))

def grade_task_3(episode_result) -> float:
    # Task 3 is highly volatile, so we use wider bounds for profit normalization
    # -5000 is a deep loss (poor management), 5000 is excellent (SOTA efficiency)
    profit_score = normalize(episode_result.net_profit, min_val=-5000, max_val=5000)
    service_score = episode_result.service_level
    crisis_score = episode_result.avg_crisis_response_score
    bullwhip_score = 1.0 - normalize(episode_result.bullwhip_coefficient, min_val=1.0, max_val=4.0)

    # Balanced SOTA weighting: 
    # 25% Profit, 25% Service, 25% Crisis Response, 10% Bullwhip Stability, 15% Strategic Reasoning
    base_score = (
        0.25 * profit_score +
        0.25 * service_score +
        0.25 * crisis_score +
        0.10 * bullwhip_score +
        0.15 * episode_result.avg_reasoning_score
    )

    return strict_clamp(round(base_score, 4))


def simulate_no_action(crisis_event, outcome) -> float:
    """Mock counterfactual simulator. Just returns outcome's net profit - some penalty."""
    return outcome.net_profit - 500.0

OPTIMAL_CRISIS_ACTIONS = {
    "supplier_shutdown": "find_alternative",
    "demand_spike": "none",
    "expiring_stock": "discount_expiring",
    "budget_freeze": "none",
    "logistics_delay": "none"
}

def grade_crisis_response(crisis_event: dict, agent_action, outcome) -> float:
    """
    Grades how appropriately the agent responded to the crisis.
    Returns 0.0 to 1.0
    """
    if not isinstance(agent_action.reasoning, str):
        reasoning = ""
    else:
        reasoning = agent_action.reasoning.lower()

    crisis_acknowledged = crisis_event["id"] in reasoning or \
                         any(kw in reasoning for kw in ["crisis", "shortage", "spike", "freeze", "expir"])

    optimal_action = OPTIMAL_CRISIS_ACTIONS.get(crisis_event["id"], "none")
    action_correct = agent_action.emergency_action == optimal_action

    counterfactual_score = simulate_no_action(crisis_event, outcome)
    # Avoid division by zero
    outcome_improvement = (outcome.net_profit - counterfactual_score) / max(1.0, abs(counterfactual_score + 1))

    return float(round(
        0.20 * float(crisis_acknowledged) +
        0.40 * float(action_correct) +
        0.40 * min(1.0, max(0.0, outcome_improvement))
    , 4))
