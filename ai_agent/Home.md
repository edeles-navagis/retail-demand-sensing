# AI Agent Wiki: Demand Sensing Orchestrator

## 📌 What the Agent Does (Summary)
The `ai_agent` is a fully-typed, standalone microservice functioning as the executive strategic intelligence engine for a retail franchise network. It handles hyper-local demand forecasting by coordinating a **two-step prediction pipeline**:
1. **Local ONNX Prediction**: Quickly computes the multi-hour (1h, 2h, 3h) demand baseline using a compiled LightGBM model (`agent_demand_multi.onnx`).
2. **LLM Demand Sensing Orchestration**: Uses the Google Gemini model (via `@google/adk`) to perform complex reasoning. It translates raw anomalies, weather, and events into actionable financial justifications (e.g., calculating revenue loss risk and recommending precise replenishment orders).

## 🔄 Processing Flow
The `DemandSensingOrchestrator` governs the lifecycle of every prediction request:
1. **Data Ingestion**: Receives the raw contextual payload (`AgentInputData`) from the backend.
2. **Local Inference (ONNX)**: The input data is mapped into mathematical feature vectors. The local ONNX runtime evaluates these features to inject multi-horizon predicted demands (`predicted_demand_1h`, `predicted_demand_2h`, `predicted_demand_3h`) into every inventory item.
3. **Token Minimization (Filtering)**: To optimize LLM token usage and focus reasoning, the orchestrator filters out items with normal demand, keeping only products experiencing a significant surge (where predicted demand > 120% of base demand).
4. **LLM Execution & Resiliency**: The filtered, high-priority data is sent to the Gemini model. The orchestrator handles API quota limits, transient errors (429/503), and performs exponential backoff retries automatically.
5. **Structured Parsing**: The LLM agent returns a strict JSON response containing the impact analysis, reasoning logs, and exact replenishment quantities, which the orchestrator validates and returns to the caller.

## 📥 Inputs to the Demand Sensing Agent
The agent accepts a highly structured JSON payload (`AgentInputData`) consisting of:
- **`store`**: Store context (UUID, name, classification, geographical profile like 'Transit BPO zone').
- **`environmental_signals`**: Real-time telemetry (temperature, heat index, UV index, precipitation).
- **`events`**: Local calendar factors (event names, dates, expected crowd sizes, paydays, holidays like Holy Week).
- **`inventory`**: Real-time stock levels, current base demand, and pricing (`product_dna`) per SKU.
- **`warehouses`**: Nearby fulfillment centers and expected travel times in minutes.

## 🧠 Inputs to the Demand Reasoning Log
When the LLM formulates its `reasoning_log` for the store owner, it strictly enforces an executive syntax format combining three distinct inputs to trace its logic:

**Format:** `"[Trigger: ...] -> [Grounding: ...] -> [Logic] ..."`

1. **Trigger (API Component Variables)**: The anomaly detected from the input streams (e.g., Weather API reporting Heat Index 42°C, Event API flagging a Mactan Fiesta or Payday).
2. **Grounding (Local Context)**: The specific geographical and demographic reality of the store (e.g., "Transit BPO zone", "Cebu Coastal Hub").
3. **Logic (Financial Impact Analysis)**: 
   - Cross-referencing triggers against 9 strict **Hyperlocal Domain Archetypes** (e.g., Solar Spikes, Typhoons, Rush Hour, Brownout Preparation).
   - Computing the inventory burn rate window based on current stock.
   - Formulating the **Revenue Loss Risk** (Potential Stockout Volume × Price).
   - Detailing the **Capital Outlay** needed for replenishment and the expected **Profit Margin** within a given timeframe.
