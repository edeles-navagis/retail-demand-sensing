import express from 'express';
import { runDemandSensing } from './agent.js';
import { AgentInputData } from './types/index.js';
import { AgentInputDataSchema } from './types/schema.js';
import dotenv from 'dotenv';
import path from 'path';

// Load environment variables (optional, mostly useful for local testing)
dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });
dotenv.config({ path: path.resolve(process.cwd(), '.env') });

const app = express();

// Vertex AI Endpoints send prediction data in JSON
app.use(express.json());

// PORT is injected by Vertex AI Custom Container environment (default 8080)
const PORT = process.env.AIP_HTTP_PORT || process.env.PORT || 8080;

/**
 * Health Check Route
 * Required by Vertex AI Endpoint custom containers.
 */
app.get('/health', (req, res) => {
  res.status(200).send('OK');
});

/**
 * Prediction Route
 * Required by Vertex AI Endpoint custom containers.
 * The payload structure Vertex AI sends is typically:
 * {
 *   "instances": [ ... ]
 * }
 */
app.post('/predict', async (req, res) => {
  try {
    const instances = req.body.instances;
    if (!instances || !Array.isArray(instances) || instances.length === 0) {
      return res.status(400).json({ error: 'Invalid input: expected non-empty "instances" array' });
    }

    // In this example, we assume we process the first instance (or loop through them)
    // The worker sends one store's context at a time via the instances array.
    const rawInput = instances[0];
    const validationResult = AgentInputDataSchema.safeParse(rawInput);

    if (!validationResult.success) {
      console.warn(
        `[AI Agent] Validation failed for payload. Errors:`,
        validationResult.error.format()
      );
      return res.status(400).json({
        error: 'Validation Error',
        details: validationResult.error.format()
      });
    }

    const agentInput = validationResult.data;

    console.log(
      `[AI Agent] Received prediction request for store ${agentInput.store?.id || 'Unknown'}`
    );

    // Call the original agent logic
    const result = await runDemandSensing(agentInput);

    // Vertex AI Endpoints expect the response in a "predictions" array
    res.status(200).json({
      predictions: [result]
    });
  } catch (err: unknown) {
    console.error('[AI Agent] Error processing prediction:', err);
    const msg = err instanceof Error ? err.message : 'Internal Server Error';
    res.status(500).json({ error: msg });
  }
});

app.listen(PORT, () => {
  console.log(`[AI Agent] Vertex AI Custom Container Server listening on port ${PORT}`);
});

// Keep event loop alive
setInterval(() => {}, 1000 * 60 * 60);
