---
title: "InventoryAI-v1 - Meta OpenEnv SOTA Benchmark"
emoji: "📦"
colorFrom: "blue"
colorTo: "pink"
sdk: "docker"
pinned: false
app_port: 7860
tags: ["openenv", "supply-chain", "reinforcement-learning"]
---

# InventoryAI-v1: A SOTA Supply Chain Benchmark for OpenEnv

**InventoryAI-v1** is a high-fidelity inventory management environment designed to evaluate the strategic reasoning and economic foresight of LLM agents.

## 🌍 Strategic Motivation: Beyond "Toy" Logistics
In the real world, inventory management is a **$2.5 Trillion** challenge. Most existing benchmarks treat this as a simple math problem. **InventoryAI-v1** models the complex, non-linear dynamics that human warehouse managers face daily:

- **The Bullwhip Effect**: Erratic ordering signals that amplify waste across the supply chain. Our environment penalizes high variance in procurement, rewarding "smooth" logistical flows.
- **Inventory Perishability (FIFO)**: Simulates the **Cost of Carry** and **Obsolescence Risk** through a batch-based FIFO (First-In-First-Out) expiration system.
- **Dynamic Price Elasticity**: Agents must rationalize prices to clear inventory before expiration, modeling the real-world trade-off between margin and volume.
- **Proactive Market Intelligence**: Foreshadows logistical crises, testing whether agents are purely reactive or capable of genuine strategic foresight.

## 🕹️ Spaces & Interface


### Observation Space (`WarehouseObservation`)
The agent receives a rich state including:
- `current_inventory`: Total units across all stock batches.
- `inventory_batches`: Detailed list of `Batch` objects with individual expiry days.
- `intelligence_feed`: Market rumors and foreshadowing hints (Proactive vs. Reactive play).
- `demand_forecast_tomorrow`: Predicted baseline demand.

### Action Space (`WarehouseAction`)
- `order_quantity`: [0-200] Units to procure.
- `sales_price`: [5.0-50.0] Daily selling price (directly affects realized demand).
- `supplier_id`: [1-3] Choose between Cheap/Slow, Balanced, or Fast/Expensive.
- `reasoning`: A required natural language justification.

## 🎯 Challenges & Tasks
1.  **Safety First (Easy)**: Focus on maintaining a 95%+ Service Level.
2.  **Profit Hero (Medium)**: Maximize net profit within a strict $5,000 procurement budget.
3.  **SOTA Crisis Manager (Hard)**: Survival during port strikes and price wars with perishable stock.

## 📊 Baseline Reference Scores
Measured over 50-day episodes using **SOTA LLMs (e.g., Llama 3.1 / GPT-4o-mini)**:

### Final Verified Baseline (Gemini 1.5 Flash)

The following scores were verified in a clean-environment 150-day simulation run:

| Task | Category | Score | Result |
| :--- | :--- | :--- | :--- |
| **Task 1** | Routine Inventory | **1.00** | ✅ Perfect |
| **Task 2** | Perishable Logistics | **0.81** | ✅ Optimized |
| **Task 3** | **Crisis Management** | **0.94** | 🏆 **SOTA** |

**Verification Log:** Refer to `testamit.txt` for the full decision-by-decision reasoning log.

## 🚀 Getting Started

### 1. Build & Run with Docker (Recommended)
```bash
docker build -t inventory-ai .
docker run -p 7860:7860 -e HF_TOKEN="your-token" inventory-ai
```

### 2. Run OpenAI Baseline (Hackathon Requirement)
Ensure your environment is running (`uvicorn server.app:app`) and run the benchmark:
```bash
export MODEL_NAME="gpt-4o-mini"
export API_KEY="sk-..."
python inference.py
```

### 3. Final Pre-Submission Validation
Run the official validator script to ensure your Space, Docker build, and OpenEnv spec are 100% compliant before submitting:
```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://your-hf-space-url.hf.space
```

## ✅ Requirements Compliance
- [x] **HF Space Ready**: Docker-based with `openenv` tag and port `7860`.
- [x] **Containerized**: Tested `Dockerfile` with multi-step execution.
- [x] **Detailed Docs**: Full motivation, space definitions, and baseline scores.
- [x] **OpenEnv Spec**: Full Pydantic/FastAPI alignment.
- [x] **Mandatory Inference**: `inference.py` follows strict `[START]/[STEP]/[END]` logging.





## 👥 Authors
- **Amit and Rishabh**

