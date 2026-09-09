import * as ort from 'onnxruntime-node';
import path from 'path';
import { fileURLToPath } from 'url';
import { ModelInstance } from '../types/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let session: ort.InferenceSession | null = null;

async function getSession() {
  if (!session) {
    const modelPath = path.join(__dirname, 'agent_demand_multi.onnx');
    session = await ort.InferenceSession.create(modelPath);
  }
  return session;
}

export async function predictLocally(instances: ModelInstance[]): Promise<number[][]> {
  if (instances.length === 0) {
    return [];
  }

  const session = await getSession();
  const batchSize = instances.length;

  const categorical_cols = ["store_id", "product_id", "category", "holiday_type"];
  
  const feeds: Record<string, ort.Tensor> = {};
  const inputNames = session.inputNames;
  
  for (const featureName of inputNames) {
    const isCat = categorical_cols.includes(featureName);
    
    // Map ONNX input names to ModelInstance keys if not directly provided
    let aliasKey = featureName;
    if (featureName === 'forecasted_temperature') aliasKey = 'temperature';
    if (featureName === 'forecasted_rainfall') aliasKey = 'rainfall';
    if (featureName === 'forecasted_heat_index') aliasKey = 'heat_index';
    if (featureName === 'foot_traffic_lag_2h') aliasKey = 'foot_traffic';

    if (isCat) {
      const data = new Array<string>(batchSize);
      for (let i = 0; i < batchSize; i++) {
        let val = (instances[i] as any)[featureName] ?? (instances[i] as any)[aliasKey];
        if (val === undefined || val === null) val = "Unknown";
        data[i] = String(val);
      }
      feeds[featureName] = new ort.Tensor('string', data, [batchSize, 1]);
    } else {
      const data = new Float32Array(batchSize);
      for (let i = 0; i < batchSize; i++) {
        let val = Number((instances[i] as any)[featureName] ?? (instances[i] as any)[aliasKey]);
        if (isNaN(val)) val = 0;
        data[i] = val;
      }
      feeds[featureName] = new ort.Tensor('float32', data, [batchSize, 1]);
    }
  }

  console.log('inputs to demand prediction model ', JSON.stringify(feeds, null, 2));

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputTensor = results[outputName];
  const data = outputTensor.data as Float32Array;

  console.log('outputs from demand prediction model ', JSON.stringify(data, null, 2));

  const predictions: number[][] = [];
  for (let i = 0; i < batchSize; i++) {
    const baseIdx = i * 3;
    predictions.push([
      Math.max(0, data[baseIdx]),
      Math.max(0, data[baseIdx + 1]),
      Math.max(0, data[baseIdx + 2])
    ]); 
  }

  return predictions;
}
