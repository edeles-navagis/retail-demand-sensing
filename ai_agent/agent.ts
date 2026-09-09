import { LlmAgent, InMemoryRunner, toStructuredEvents, EventType } from '@google/adk';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { predictLocally } from './predictor/localPredictor.js';
import { activeSessionsData } from './store/sessionStore.js';
import { getAgentTools } from './tools/tools.js';
import { SYSTEM_INSTRUCTION } from './prompts/prompts.js';
import { mapInventoryToInstances } from './utils/mapper.js';
import { AgentInputData, AgentResponse, InventoryItem } from './types/index.js';
import { exit } from 'process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, '.env.local') });
dotenv.config({ path: path.join(__dirname, '.env') });

export const demandSensingAgent = new LlmAgent({
  name: 'DemandSensingAgent',
  model: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
  description: 'AI Reasoning agent for demand change prediction',
  instruction: SYSTEM_INSTRUCTION,
  generateContentConfig: {
    httpOptions: { retryOptions: { attempts: 1 } }
  }
});

/**
 * Agentic Workflow Orchestrator
 */
class DemandSensingOrchestrator {
  private static readonly MAX_ITERATIONS = 10;
  private static readonly MAX_RETRIES = 3;

  static async run(inputData: AgentInputData): Promise<AgentResponse> {
    let retryCount = 0;

    // Call the demand predictor model BEFORE execution so it is populated for the agent
    try {
      if (inputData.inventory && inputData.inventory.length > 0) {
        const instances = mapInventoryToInstances(inputData);

        try {
          console.log(
            `[Orchestrator] Evaluating expected demand model locally inside ai_agent for ${instances.length} items...`
          );
          const predictions = await predictLocally(instances);
          for (let i = 0; i < inputData.inventory.length; i++) {
            const pred = predictions[i] ?? [0, 0, 0];
            inputData.inventory[i].predicted_demand_1h = parseFloat((pred[0]).toFixed(2));
            inputData.inventory[i].predicted_demand_2h = parseFloat((pred[1]).toFixed(2));
            inputData.inventory[i].predicted_demand_3h = parseFloat((pred[2]).toFixed(2));
          }
          console.log(
            `[Orchestrator] Successfully injected predicted 1h, 2h, 3h demand locally for ${inputData.inventory.length} items.`
          );
        } catch (err: unknown) {
          const errorMessage = err instanceof Error ? err.message : String(err);
          console.error(
            `[Orchestrator] Local prediction failed: ${errorMessage}. Using fallback historical demands.`
          );
          for (const inv of inputData.inventory) {
            inv.predicted_demand_1h = inv.historical_sales_t_minus_3 ?? inv.base_demand;
            inv.predicted_demand_2h = inv.historical_sales_t_minus_2 ?? inv.base_demand;
            inv.predicted_demand_3h = inv.historical_sales_t_minus_1 ?? inv.base_demand;
          }
        }
      }
    } catch (outerErr: unknown) {
      const outerErrorMessage = outerErr instanceof Error ? outerErr.message : String(outerErr);
      console.error(
        `[Orchestrator] Unexpected error in local prediction pipeline: ${outerErrorMessage}. Using fallback historical demands.`
      );
      if (inputData.inventory) {
        for (const inv of inputData.inventory) {
          inv.predicted_demand_1h = inv.historical_sales_t_minus_3 ?? inv.base_demand;
          inv.predicted_demand_2h = inv.historical_sales_t_minus_2 ?? inv.base_demand;
          inv.predicted_demand_3h = inv.historical_sales_t_minus_1 ?? inv.base_demand;
        }
      }
    }

    const calculateRunwayInHours = (
      currentStock: number,
      hourlyDemands: readonly number[]
    ): number => {
      if (currentStock <= 0) return 0;

      let stockLeft = currentStock;
      let totalHours = 0;

      for (const demand of hourlyDemands) {
        // Prevent zero/negative demand from creating infinite loops or division by zero
        if (demand <= 0) continue;

        if (stockLeft >= demand) {
          stockLeft -= demand;
          totalHours += 1;
        } else {
          totalHours += stockLeft / demand;
          stockLeft = 0;
          break;
        }
      }

      // Extrapolate at last known burn rate if stock outlasts provided hourly demand array
      if (stockLeft > 0 && hourlyDemands.length > 0) {
        const lastDemand = hourlyDemands[hourlyDemands.length - 1];
        if (lastDemand > 0) {
          totalHours += stockLeft / lastDemand;
        }
      }

      const totalMinutes = Math.round(totalHours * 60);

      return Number(totalHours.toFixed(2));
    }

    // Deterministic Planning and Truck Packing Phase
    if (inputData.inventory && inputData.warehouses && inputData.warehouses.length > 0) {
      const bestWh = Object.assign({}, [...inputData.warehouses].sort((a, b) => (a.travel_time_mins || 999) - (b.travel_time_mins || 999))[0]);
      (inputData as any).bestWh = bestWh;
      const safetyBufferFactorGlobal = 0.25;
      const productBufferFactors = inputData.settings?.product_buffer_factors || [];
      const factorMap = new Map<string, number>();
      for (const pbf of productBufferFactors) {
        factorMap.set(pbf.category.toLowerCase().trim(), pbf.buffer_factor);
      }

      const critHours = inputData.settings?.critical_hours ?? 3;
      const midHours = inputData.settings?.mid_hours ?? 24;

      delete(inputData.warehouses); // minimize token

      const priorityMap: Record<string, number> = { 'CRITICAL': 1, 'URGENT': 2, 'MID': 3, 'LOW': 4, 'SAFE': 5 };
      let computed: any[] = [];
      let criticalInv: any[] = [];

      for (const inv of inputData.inventory) {
        const d1 = inv.predicted_demand_1h ?? inv.historical_sales_t_minus_3 ?? inv.base_demand;
        const d2 = inv.predicted_demand_2h ?? inv.historical_sales_t_minus_2 ?? inv.base_demand;
        const d3 = inv.predicted_demand_3h ?? inv.historical_sales_t_minus_1 ?? inv.base_demand;
        const totalDemand = d1 + d2 + d3;
        const runway = calculateRunwayInHours(inv.current_stock, [d1, d2, d3]);

        let impact_factor = 'SAFE';
        if (runway <= critHours) impact_factor = 'CRITICAL';
        else if (runway < midHours) impact_factor = 'URGENT';
        else if (runway < midHours * 2) impact_factor = 'LOW';

        const cat = (inv.category || '').toLowerCase().trim();
        const factor = factorMap.has(cat) ? factorMap.get(cat)! : safetyBufferFactorGlobal;
        const safetyStockBuffer = ((d1 + d2 + d3) / 3) * factor;

        const restock = Math.max(0, totalDemand - inv.current_stock + safetyStockBuffer);
        const bundleQty = inv.wholesale_bundle_qty && inv.wholesale_bundle_qty > 0 ? inv.wholesale_bundle_qty : 1;
        const recommended_replenishment = Math.ceil(restock / bundleQty) * bundleQty;

        const histAvg = inv.product_historical_avg_sales ?? 0;
        const isAffectedByEventOrWeather = (totalDemand / 3) > histAvg;

        const categoryLower = (inv.category || '').toLowerCase().trim();
        let isRelevantToCurrentContext = false;

        const temp = inputData.environmental_signals?.temp ?? 32;
        const precip = inputData.environmental_signals?.precipitation ?? 0;
        const heatIndex = inputData.environmental_signals?.heat_index ?? 35;
        const month = new Date().getMonth() + 1;
        const hour_of_day = new Date().getHours();

        if (temp >= 40 || heatIndex >= 40) {
          if (['beverages', 'skin care', 'personal care', 'baking'].includes(categoryLower)) {
            isRelevantToCurrentContext = true;
          }
        }
        if (precip >= 5) {
          if (['canned goods', 'home essentials', 'electronics', 'fresh / preserved', 'health care'].includes(categoryLower)) {
            isRelevantToCurrentContext = true;
          }
        }
        if (precip > 0 && [6, 7, 8].includes(month)) {
          if (['health care'].includes(categoryLower)) {
            isRelevantToCurrentContext = true;
          }
        }
        if ((hour_of_day >= 6 && hour_of_day <= 9) || (hour_of_day >= 17 && hour_of_day <= 20)) {
          if (['snacks', 'home essentials', 'health care'].includes(categoryLower)) {
            isRelevantToCurrentContext = true;
          }
        }

        const activeEvents = (inputData.events || []).map(e => (e.event_name || e.name || '').toLowerCase());
        for (const evt of activeEvents) {
          if (evt.includes('heatwave') || evt.includes('heat') || evt.includes('solar')) {
            if (['beverages', 'skin care', 'personal care', 'baking'].includes(categoryLower)) {
              isRelevantToCurrentContext = true;
            }
          }
          if (evt.includes('typhoon') || evt.includes('cyclone') || evt.includes('storm') || evt.includes('rain')) {
            if (['canned goods', 'home essentials', 'electronics', 'fresh / preserved', 'health care'].includes(categoryLower)) {
              isRelevantToCurrentContext = true;
            }
          }
          if (evt.includes('fiesta') || evt.includes('payday') || evt.includes('holiday') || evt.includes('charter') || evt.includes('celebration')) {
            if (['beverages', 'apparel', 'baking', 'canned goods', 'fresh / preserved', 'skin care'].includes(categoryLower)) {
              isRelevantToCurrentContext = true;
            }
          }
          if (evt.includes('brownout') || evt.includes('power') || evt.includes('grid')) {
            if (['electronics'].includes(categoryLower)) {
              isRelevantToCurrentContext = true;
            }
          }
          if (evt.includes('school') || evt.includes('academic') || evt.includes('opening')) {
            if (['stationery', 'snacks'].includes(categoryLower)) {
              isRelevantToCurrentContext = true;
            }
          }
        }

        const hasActiveContext = activeEvents.length > 0 || temp >= 40 || heatIndex >= 40 || precip >= 5;
        if (!hasActiveContext) {
          isRelevantToCurrentContext = true;
        }

        if (recommended_replenishment > 0 && isAffectedByEventOrWeather && isRelevantToCurrentContext) {
          computed.push({
            product_id: inv.product_id,
            impact_factor,
            projected_demand_increase_pct: histAvg > 0 ? (((totalDemand / 3) - histAvg) / histAvg) * 100 : 0,
            recommended_replenishment,
            source_warehouse_id: bestWh.id,
            estimated_delivery_arrival_time: (bestWh.travel_time_mins || 0) + 45,
            weight: (recommended_replenishment / bundleQty) * (inv.wholesale_gross_weight_kg || 0),
            volume: (recommended_replenishment / bundleQty) * (inv.wholesale_gross_volume_m3 || 0),
            name: inv.name
          });
        }
        
        criticalInv.push({
          product_id: inv.product_id,
          current_quantity: inv.current_stock,
          expected_quantity: totalDemand,
          predicted_demand_1h: d1,
          predicted_demand_2h: d2,
          predicted_demand_3h: d3,
          name: inv.name
        });
      }

      computed.sort((a, b) => (priorityMap[a.impact_factor] || 99) - (priorityMap[b.impact_factor] || 99));
      
      const truckWeightLimit = bestWh.truck_weight_limit || 10000;
      const truckVolumeLimit = bestWh.truck_volume_limit || 100;
      let totalW = 0;
      let totalV = 0;
      let packed = [];
      
      for (const item of computed) {
        if (totalW + item.weight <= truckWeightLimit && totalV + item.volume <= truckVolumeLimit) {
          totalW += item.weight;
          totalV += item.volume;
          packed.push(item);
        }
      }

      inputData.computed_recommendations = packed;
      (inputData as any).computed_critical_inventory = criticalInv;
      console.log(`[Orchestrator] Packed ${packed.length} items into truck from ${bestWh.name}.`);
    }

    while (retryCount < this.MAX_RETRIES) {
      try {
        console.log(
          `[Orchestrator] Calling Gemini model for "${inputData.store?.name || 'Store'}" (Attempt ${retryCount + 1}/${this.MAX_RETRIES})`
        );
        return await this.execute(inputData);
      } catch (err: unknown) {
        console.error('[Orchestrator Debug] Execute failed with error:', err);
        retryCount++;
        const errorMessage = err instanceof Error ? err.message : String(err);
        const serializedError = JSON.stringify(err || '').toLowerCase();
        const errStr = (errorMessage + ' ' + serializedError).toLowerCase();
        const isTransient =
          errStr.includes('429') ||
          errStr.includes('503') ||
          errStr.includes('high demand') ||
          errStr.includes('quota') ||
          errStr.includes('exhausted') ||
          errStr.includes('rate');

        if (isTransient && retryCount < this.MAX_RETRIES) {
          // Parse the exact retry time from "Please retry in 42.736255349s." if present
          let backoff = Math.pow(2, retryCount) * 1000;
          const retryMatch = errorMessage.match(/retry in ([\d\.]+)s/i);
          if (retryMatch && retryMatch[1]) {
            backoff = (parseFloat(retryMatch[1]) * 1000) + 1000; // Add 1s buffer
          } else if (errStr.includes('quota')) {
            backoff = 22000; // Fallback quota wait
          }
          console.warn(`[Orchestrator] Transient API error. Retrying in ${Math.round(backoff / 1000)}s...`);
          await new Promise((resolve) => setTimeout(resolve, backoff));
          continue;
        }

        console.warn(`[Orchestrator] Resiliency Fallback: Returning empty response for store "${inputData.store?.name || 'Store'}" due to API constraints.`);
        return {
          store_id: inputData.store?.id || '',
          confidence_score: 0,
          reasoning_log: 'Fallback activated due to API constraints or formatting errors.',
          predictions: inputData.computed_recommendations || [],
          critical_product_inventory: (inputData as any).computed_critical_inventory || []
        } as AgentResponse;
      }
    }

    console.warn(`[Orchestrator] Retry limit exceeded. Falling back to empty response.`);
    return {
      store_id: inputData.store?.id || '',
      confidence_score: 0,
      reasoning_log: 'Fallback activated due to API constraints or formatting errors.',
      predictions: inputData.computed_recommendations || [],
      critical_product_inventory: (inputData as any).computed_critical_inventory || []
    } as AgentResponse;
  }


  private static async execute(inputData: AgentInputData): Promise<AgentResponse> {
    
    const runner = new InMemoryRunner({ agent: demandSensingAgent, appName: 'DemandSensingApp' });
    const userId = 'system-worker';
    const sessionId = `session-${Date.now()}`;

    // console.log('DemandSensingOrchestrator execute inputData', JSON.stringify(inputData, null, 2));

    await runner.sessionService.createSession({ appName: 'DemandSensingApp', userId, sessionId });

    activeSessionsData.set(sessionId, inputData);
    const bestWh = (inputData as any).bestWh || (inputData.warehouses && inputData.warehouses.length > 0 
      ? [...inputData.warehouses].sort((a, b) => (a.travel_time_mins || 999) - (b.travel_time_mins || 999))[0]
      : null);

    const truck_flat_rate = bestWh?.truck_flat_rate || 0;
    const truck_rate = bestWh?.delivery_rate || bestWh?.truck_rate || 0;
    const distance_km = bestWh?.distance_km || ((bestWh?.travel_time_mins || 0) / 2); // Fallback assumption
    const cost_of_delivery = truck_flat_rate + (truck_rate * distance_km);
    const estimated_total_delivery_time = (bestWh?.travel_time_mins || 0) + 45; // including 45m lead time

    let total_cost_of_purchase = 0;
    let total_revenue_loss_risk = 0;

    const enrichedRecommendations = (inputData.computed_recommendations || []).map(r => {
      const invMatch = inputData.inventory?.find(i => i.product_id === r.product_id);
      console.log('r.product_id', r.product_id);
      console.log('>>> invMatch', JSON.stringify(invMatch, null, 2));
      
      const qty = r.recommended_replenishment || 0;
      const wholesale = ((invMatch?.wholesale_price || 0) / (invMatch?.wholesale_bundle_qty ?? 1)) || 0;
      const price = invMatch?.price || 0;
      
      const d1 = invMatch?.predicted_demand_1h || 0;
      const d2 = invMatch?.predicted_demand_2h || 0;
      const d3 = invMatch?.predicted_demand_3h || 0;
      const sensed_demand = d1 + d2 + d3;
      const current_qty = invMatch?.current_stock || 0;
      
      const shortage = Math.max(0, sensed_demand - current_qty);
      const revenue_loss_risk = shortage * price;
      const wholesale_cost = qty * wholesale;
      
      total_cost_of_purchase += wholesale_cost;
      total_revenue_loss_risk += revenue_loss_risk;

      return {
        name: r.name,
        recommended_replenishment: qty,
        impact_factor: r.impact_factor,
        projected_demand_increase_pct: r.projected_demand_increase_pct,
        price: price,
        wholesale_price: wholesale,
        current_quantity: current_qty,
        sensed_next_hour_demand: d1,
        wholesale_capital_outlay: wholesale_cost,
        revenue_loss_risk: revenue_loss_risk,
        shortage_volume: shortage,
        burn_rate_window_hours: d1 > 0 ? parseFloat((current_qty / d1).toFixed(2)) : 999
      };
    });

    // remove those items which has 0 revenue loss risk
    const filteredRecommendations = enrichedRecommendations.filter(r => r.revenue_loss_risk > 0);

    const calculated_total_restock_cost = cost_of_delivery + total_cost_of_purchase;
    const calculated_total_risk_profit = total_revenue_loss_risk - calculated_total_restock_cost;

    console.log('\n=== [Sensed Demand Spike Log (Worker Input)] ===');
    for (const inv of inputData.inventory || []) {
      const histAvg = inv.product_historical_avg_sales ?? 0;
      const d1 = inv.predicted_demand_1h ?? 0;
      const d2 = inv.predicted_demand_2h ?? 0;
      const d3 = inv.predicted_demand_3h ?? 0;
      const avgPred = (d1 + d2 + d3) / 3;
      if (avgPred > histAvg) {
        console.log(`Product: ${inv.name} (${inv.product_id}) | Category: ${inv.category}`);
        console.log(`  Raw Current Stock: ${inv.current_stock}`);
        console.log(`  Historical Store-Product Avg Sales: ${histAvg}`);
        console.log(`  Predicted Demands: 1h=${d1}, 2h=${d2}, 3h=${d3} | Avg Sensed: ${avgPred.toFixed(2)}`);
      }
    }
    console.log('================================================\n');

    const minimizedPayload = {
      store: {
         id: inputData.store?.id,
         name: inputData.store?.name,
         classification: inputData.store?.classification,
         geographical_profile: inputData.store?.geographical_profile
      },
      environmental_signals: inputData.environmental_signals,
      events: inputData.events?.map(e => ({
        event_name: e.event_name,
        event_type: e.event_type,
        expected_crowd: e.expected_crowd
      })),
      delivery_cost: cost_of_delivery,
      total_cost_of_purchase: total_cost_of_purchase,
      calculated_total_restock_cost: calculated_total_restock_cost,
      estimated_total_delivery_time: estimated_total_delivery_time,
      calculated_total_risk_profit: calculated_total_risk_profit,
      recommendations: filteredRecommendations
    };

    console.log('minimizedPayload to gemini:', sessionId);
    console.log(JSON.stringify(minimizedPayload, null, 2));    

    const inputDataStr = JSON.stringify(minimizedPayload);
    const message = { role: 'user', parts: [{ text: inputDataStr }] };
    
    // Calculate approximate token usage
    const inputTokens = Math.ceil(inputDataStr.length / 4);
    const systemTokens = Math.ceil(SYSTEM_INSTRUCTION.length / 4);
    const totalTokens = inputTokens + systemTokens;
    console.log(`[Orchestrator] Token Usage Estimate: Input Data ~${inputTokens} tokens, System Instructions ~${systemTokens} tokens. Total ~${totalTokens} tokens.`);

    let finalResponse = '';
    let iterations = 0;

    console.log(
      `[Orchestrator] Starting inference for ${inputData.store?.name || inputData.store?.id}`
    );

    try {
      for await (const event of runner.runAsync({ userId, sessionId, newMessage: message })) {
        iterations++;

        if (iterations > this.MAX_ITERATIONS) {
          throw new Error(
            `[Orchestrator] Maximum iteration limit (${this.MAX_ITERATIONS}) reached. Potential reasoning loop.`
          );
        }

        const structuredEvents = toStructuredEvents(event);
        for (const se of structuredEvents) {
          if (se.type === EventType.CONTENT || se.type === EventType.THOUGHT) {
            finalResponse += se.content;
          } else if (se.type === EventType.ERROR) {
            console.error('[Agent Error]', se.error);

            const errObj = se.error as Error | null | undefined;
            const errorMessage = errObj?.message || String(se.error);
            const serializedError = JSON.stringify(se.error || '').toLowerCase();
            const errStr = (errorMessage + ' ' + serializedError).toLowerCase();
            const isTransient =
              errStr.includes('429') ||
              errStr.includes('503') ||
              errStr.includes('high demand') ||
              errStr.includes('quota') ||
              errStr.includes('exhausted') ||
              errStr.includes('rate');

            if (!isTransient) {
              throw new Error(`Agent encountered a non-transient error: ${errorMessage}`);
            }
            throw se.error; // Trigger transient retry
          }
        }
      }
    } finally {
      activeSessionsData.delete(sessionId);
    }

    return this.parseResponse(finalResponse, inputData);
  }

  private static parseResponse(raw: string, inputData: AgentInputData): AgentResponse {
    if (!raw) throw new Error('Agent returned empty response');

    try {
      // Remove scratchpad if present
      const cleanRaw = raw.replace(/<scratchpad>[\s\S]*?<\/scratchpad>/g, '').trim();
      
      // Robust JSON extraction
      const jsonMatch = cleanRaw.match(/```(?:json)?\n([\s\S]*?)\n```/) || cleanRaw.match(/(\{[\s\S]*\})/);
      let jsonStr = jsonMatch ? jsonMatch[1] : cleanRaw;
      
      if (!jsonStr.trim().startsWith('{')) {
          jsonStr = JSON.stringify({
              store_id: inputData.store?.id || '',
              confidence_score: 0.5,
              reasoning_log: jsonStr.trim()
          });
      }

      let parsed: any;
      try {
          parsed = JSON.parse(jsonStr.trim());
      } catch (parseErr) {
          // If JSON parsing fails due to unescaped quotes or newlines,
          // treat the entire output as the reasoning log to prevent a crash.
          parsed = {
              store_id: inputData.store?.id || '',
              confidence_score: 0.5,
              reasoning_log: cleanRaw
          };
      }

      // Basic schema validation
      if (!parsed.store_id) {
        parsed.store_id = inputData.store?.id || '';
      }

      const response = parsed as AgentResponse;

      // Filter predictions to only include recommendations with revenue_loss_risk > 0, matching minimizedPayload
      const enriched = (inputData.computed_recommendations || []).map(r => {
        const invMatch = inputData.inventory?.find(i => i.product_id === r.product_id);
        const price = invMatch?.price || 0;
        const d1 = invMatch?.predicted_demand_1h || 0;
        const d2 = invMatch?.predicted_demand_2h || 0;
        const d3 = invMatch?.predicted_demand_3h || 0;
        const sensed_demand = d1 + d2 + d3;
        const current_qty = invMatch?.current_stock || 0;
        const shortage = Math.max(0, sensed_demand - current_qty);
        const revenue_loss_risk = shortage * price;
        return {
          ...r,
          revenue_loss_risk
        };
      });
      response.predictions = enriched.filter(r => r.revenue_loss_risk > 0).map(r => {
        const { revenue_loss_risk, ...rest } = r;
        return rest;
      });

      response.critical_product_inventory = (inputData as any).computed_critical_inventory || [];

      // Reconstruct minimizedPayload for the association log
      const logBestWh = (inputData as any).bestWh || (inputData.warehouses && inputData.warehouses.length > 0 
        ? [...inputData.warehouses].sort((a, b) => (a.travel_time_mins || 999) - (b.travel_time_mins || 999))[0]
        : null);
      const logTruckFlatRate = logBestWh?.truck_flat_rate || 0;
      const logTruckRate = logBestWh?.delivery_rate || logBestWh?.truck_rate || 0;
      const logDistanceKm = logBestWh?.distance_km || ((logBestWh?.travel_time_mins || 0) / 2);
      const logCostOfDelivery = logTruckFlatRate + (logTruckRate * logDistanceKm);
      let logTotalCostOfPurchase = 0;
      let logTotalRevenueLossRisk = 0;

      const logEnrichedRecommendations = (inputData.computed_recommendations || []).map(r => {
        const invMatch = inputData.inventory?.find(i => i.product_id === r.product_id);
        const price = invMatch?.price || 0;
        const d1 = invMatch?.predicted_demand_1h || 0;
        const d2 = invMatch?.predicted_demand_2h || 0;
        const d3 = invMatch?.predicted_demand_3h || 0;
        const sensed_demand = d1 + d2 + d3;
        const current_qty = invMatch?.current_stock || 0;
        const shortage = Math.max(0, sensed_demand - current_qty);
        const revenue_loss_risk = shortage * price;
        const qty = r.recommended_replenishment || 0;
        const wholesale = ((invMatch?.wholesale_price || 0) / (invMatch?.wholesale_bundle_qty ?? 1)) || 0;
        logTotalCostOfPurchase += qty * wholesale;
        logTotalRevenueLossRisk += revenue_loss_risk;
        return {
          name: r.name,
          recommended_replenishment: qty,
          impact_factor: r.impact_factor,
          projected_demand_increase_pct: r.projected_demand_increase_pct,
          price,
          wholesale_price: wholesale,
          current_quantity: current_qty,
          sensed_next_hour_demand: d1,
          wholesale_capital_outlay: qty * wholesale,
          revenue_loss_risk,
          shortage_volume: shortage,
          burn_rate_window_hours: d1 > 0 ? parseFloat((current_qty / d1).toFixed(2)) : 999
        };
      });

      const logMinimizedPayload = {
        store: {
           id: inputData.store?.id,
           name: inputData.store?.name,
           classification: inputData.store?.classification,
           geographical_profile: inputData.store?.geographical_profile
        },
        environmental_signals: inputData.environmental_signals,
        events: inputData.events?.map(e => ({
          event_name: e.event_name,
          event_type: e.event_type,
          expected_crowd: e.expected_crowd
        })),
        delivery_cost: logCostOfDelivery,
        total_cost_of_purchase: logTotalCostOfPurchase,
        calculated_total_restock_cost: logCostOfDelivery + logTotalCostOfPurchase,
        estimated_total_delivery_time: (logBestWh?.travel_time_mins || 0) + 45,
        calculated_total_risk_profit: logTotalRevenueLossRisk - (logCostOfDelivery + logTotalCostOfPurchase),
        recommendations: logEnrichedRecommendations.filter(r => r.revenue_loss_risk > 0)
      };

      console.log('\n=== [Mini JSON Input to Actual Reasoning & Recommended Restocks] ===');
      console.log('--- Mini JSON Input ---');
      console.log(JSON.stringify(logMinimizedPayload, null, 2));
      console.log('--- Actual Reasoning Log ---');
      console.log(response.reasoning_log);
      console.log('--- Recommended Restocks ---');
      if (response.predictions && response.predictions.length > 0) {
        for (const pred of response.predictions) {
          const name = inputData.inventory?.find(i => i.product_id === pred.product_id)?.name || 'Unknown';
          console.log(`  Product: ${name} (${pred.product_id}) | Recommended Qty: ${pred.recommended_replenishment} | Reason: ${pred.impact_factor}`);
        }
      } else {
        console.log('  No restocks recommended.');
      }
      console.log('===================================================================\n');

      // Calculate cost_of_delivery using bestWh
      const bestWh = (inputData as any).bestWh || (inputData.warehouses && inputData.warehouses.length > 0 
        ? [...inputData.warehouses].sort((a, b) => (a.travel_time_mins || 999) - (b.travel_time_mins || 999))[0]
        : null);

      const truck_flat_rate = bestWh?.truck_flat_rate || 0;
      const truck_rate = bestWh?.delivery_rate || bestWh?.truck_rate || 0;
      const distance_km = bestWh?.distance_km || ((bestWh?.travel_time_mins || 0) / 2);
      const cost_of_delivery = truck_flat_rate + (truck_rate * distance_km);

      // Calculate total_cost_of_purchase
      let total_cost_of_purchase = 0;
      for (const pred of response.predictions) {
        const invMatch = inputData.inventory?.find(i => i.product_id === pred.product_id);
        if (invMatch) {
          const qty = pred.recommended_replenishment || 0;
          const wholesale = ((invMatch.wholesale_price || 0) / (invMatch.wholesale_bundle_qty ?? 1)) || 0;
          total_cost_of_purchase += qty * wholesale;
        }
      }

      // Calculate total_risk_cost
      let total_risk_cost = 0;
      for (const inv of inputData.inventory || []) {
        const d1 = inv.predicted_demand_1h || 0;
        const d2 = inv.predicted_demand_2h || 0;
        const d3 = inv.predicted_demand_3h || 0;
        const totalDemand3h = d1 + d2 + d3;
        const currentStock = inv.current_stock || 0;
        const shortage = Math.max(0, totalDemand3h - currentStock);
        const retailPrice = inv.price || 0;
        const wholesalePricePerPcs = ((inv.wholesale_price || 0) / (inv.wholesale_bundle_qty ?? 1)) || 0;
        const margin = retailPrice - wholesalePricePerPcs;
        const riskCost = shortage * margin;
        total_risk_cost += riskCost;
      }

      const budgetLimit = inputData.settings?.budget_limit_php ?? 100000;
      const confidence = typeof response.confidence_score === 'number' ? response.confidence_score : parseFloat(response.confidence_score as any) || 0;

      // Auto-approve unless any condition is met
      let autoApprove = true;

      if (total_cost_of_purchase + cost_of_delivery > budgetLimit) {
        console.log(`[Approval Gate] MANUAL_APPROVAL triggered: Total Cost (${total_cost_of_purchase + cost_of_delivery}) > Budget Limit (${budgetLimit})`);
        autoApprove = false;
      } else if (confidence < 0.8) {
        console.log(`[Approval Gate] MANUAL_APPROVAL triggered: Confidence Score (${confidence}) < 80%`);
        autoApprove = false;
      } else if (cost_of_delivery > total_risk_cost) {
        console.log(`[Approval Gate] MANUAL_APPROVAL triggered: Cost of Delivery (${cost_of_delivery}) > Total Risk Cost (${total_risk_cost})`);
        autoApprove = false;
      }

      response.approval_status = autoApprove ? 'AUTO_APPROVED' : 'MANUAL_APPROVAL';

      return response;
    } catch (e: unknown) {
      console.error('[Orchestrator] Response parsing failed:', raw);
      const msg = e instanceof Error ? e.message : String(e);
      throw new Error(`Failed to parse agent response: ${msg}`);
    }
  }
}

/**
 * Main entry point for the worker
 */
export async function runDemandSensing(inputData: AgentInputData): Promise<AgentResponse> {
  return await DemandSensingOrchestrator.run(inputData);
}
