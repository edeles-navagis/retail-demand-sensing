# Spark AI Agent: Demand Sensing Orchestrator

The `ai_agent` module is a fully-typed, standalone microservice designed to handle hyper-local demand forecasting. It operates as a Custom Container deployed on **Google Vertex AI Endpoints** and coordinates a two-step prediction pipeline:

1. **Local Heuristic / ONNX Prediction**: Fast, immediate prediction of next-hour demand utilizing an ONNX Runtime node environment.
2. **LLM Demand Sensing Orchestration**: Complex decision-making and reasoning processed through the Gemini model via the Google Agent SDK (`@google/adk`).

## 📁 Architecture

- **`server.ts`**: Express application entrypoint. Provides `/health` and `/predict` routes specifically formatted to adhere to the Vertex AI Endpoint contract.
- **`agent.ts`**: The core LLM agent orchestration logic utilizing `@google/adk`. Handles Gemini interactions, resiliency, retries, and API quota limits.
- **`predictor/localPredictor.ts`**: In-memory rapid predictor utilizing ONNX runtime.
- **`types/index.ts` & `types/schema.ts`**: Strict TypeScript interfaces and corresponding `zod` schemas enforcing rigid validation on incoming data.
- **`tests/`**: Automated test suites running on **Vitest**.

## 🚀 Getting Started

### Prerequisites

- Node.js (v20+)
- Access to Google Gemini API (Ensure your `.env` contains the required credentials)

### Installation

Navigate to the `ai_agent` directory and install dependencies:

```bash
npm install
```

### Running Locally

To start the local development server on port 8080:

```bash
npm run start
```

### Environment Variables

Create a `.env` or `.env.local` file at the root of `ai_agent`:

```env
# Required for @google/adk interactions with Gemini
GEMINI_API_KEY="your_api_key_here"
```

## 🛠 Scripts and Tooling

This module enforces strict code quality and formatting.

- **`npm run test`**: Executes the Vitest test suites.
- **`npm run lint`**: Runs ESLint v9 checks across the codebase.
- **`npm run format`**: Formats the codebase using Prettier.
- **`npm run deploy:vertex`**: Helper command outline for deploying the customized container to Google Cloud Artifact Registry and Vertex AI.

## 📡 API Contract

The prediction payload expects Vertex AI standardized JSON formatting (`{ "instances": [ ... ] }`).

### Example POST `/predict`
```json
{
  "instances": [
    {
      "store": {
        "id": "53f4af12-5ad2-43f7-b8b6-d00d8686fa2c",
        "name": "Spark MEPZ 1",
        "classification": "Industrial"
      },
      "environmental_signals": {
        "temp": 39,
        "heat_index": 42
      },
      "inventory": [
        {
          "product_id": "SKU-BEV-BI-009",
          "current_stock": 10,
          "base_demand": 100
        }
      ]
    }
  ]
}
```

The system will validate this input dynamically via `zod`. Malformed requests will immediately return a `400 Bad Request` with detailed error traces.
