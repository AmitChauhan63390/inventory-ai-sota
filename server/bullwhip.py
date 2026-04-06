import numpy as np
from typing import List

def calculate_bullwhip_coefficient(order_history: List[int], demand_history: List[int]) -> float:
    """
    Bullwhip coefficient = (std_dev of orders / mean of orders) / (std_dev of demand / mean of demand)
    
    Perfect agent = coefficient close to 1.0 (orders vary proportionally with demand)
    Panicking agent = coefficient >> 1.0 (orders swing much more than demand)
    
    Returns coefficient. Lower is better. Cap display at 5.0 for readability.
    """
    if len(order_history) < 3 or len(demand_history) < 3:
        return 1.0  # not enough data

    order_cv = np.std(order_history) / (np.mean(order_history) + 1e-9)
    demand_cv = np.std(demand_history) / (np.mean(demand_history) + 1e-9)

    coefficient = order_cv / (demand_cv + 1e-9)
    return float(round(min(5.0, coefficient), 4))
