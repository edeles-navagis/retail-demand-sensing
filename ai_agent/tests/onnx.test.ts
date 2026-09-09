import { describe, it, expect } from 'vitest';
import { predictLocally } from '../predictor/localPredictor.js';

describe('ONNX Local Predictor', () => {
  it('should successfully execute prediction on dummy instances', async () => {
    const dummyInstances = [
      {
        store_id: 'store_123',
        product_id: 'prod_456',
        category: 'Beverage',
        price: 50,
        inventory: 20,
        promo_flag: 0,
        payday_flag: 1,
        weekend_flag: 0,
        holiday_type: 'No Holiday',
        nearby_event_flag: 0,
        stockout_flag: 0,
        brownout_flag: 0,
        temperature: 32,
        rainfall: 0,
        heat_index: 38,
        hour_of_day: 14,
        day_of_week: 3,
        month: 6,
        qty_sold_lag_24h: 3,
        foot_traffic_lag_2h: 100,
        store_historical_avg_sales: 50,
        product_historical_avg_sales: 10,
        rolling_mean_demand_3h: 12,
        rolling_mean_demand_6h: 12,
        rolling_mean_demand_24h: 12,
        rolling_std_6h: 2
      }
    ];

    const predictions = await predictLocally(dummyInstances);

    expect(predictions).toBeDefined();
    expect(predictions).toBeInstanceOf(Array);
    expect(predictions.length).toBe(1);
    expect(predictions[0].length).toBe(3);
    expect(typeof predictions[0][0]).toBe('number');
    expect(typeof predictions[0][1]).toBe('number');
    expect(typeof predictions[0][2]).toBe('number');
  }, 30000);

  it('should return empty array when given no instances', async () => {
    const predictions = await predictLocally([]);
    expect(predictions).toEqual([]);
  });
});
