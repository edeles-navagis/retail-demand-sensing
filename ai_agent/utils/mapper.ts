import {
  AgentInputData,
  ModelInstance,
  InventoryItem,
  EventData
} from '../types/index.js';

export function mapInventoryToInstances(inputData: AgentInputData): ModelInstance[] {
  if (!inputData.inventory || inputData.inventory.length === 0) {
    return [];
  }

  return inputData.inventory.map((inv: InventoryItem): ModelInstance => {
    const productId = inv.product_id;
    const category = inv.category || 'Beverages';

    const hasEvents = inputData.events && inputData.events.length > 0;
    const expectedCrowds =
      hasEvents && inputData.events
        ? inputData.events.map((e: EventData) => {
            const val = e.expected_crowd || e.crowd;
            return typeof val === 'string' ? parseInt(val) : typeof val === 'number' ? val : 0;
          })
        : [];
    const expectedCrowd =
      expectedCrowds.length > 0 ? Math.max(...expectedCrowds.filter((n) => !isNaN(n))) : 0;
    const foot_traffic = expectedCrowd > 5000 ? 300 : expectedCrowd > 0 ? 200 : 120;

    const promo_flag =
      inputData.events &&
      inputData.events.some((e: EventData) => {
        const eType = (e.event_type || e.type || '').toLowerCase();
        const eName = (e.event_name || e.name || '').toLowerCase();
        return eType === 'sale' || eName.includes('sale') || eName.includes('promo');
      })
        ? 1
        : 0;

    const payday_flag =
      inputData.events &&
      inputData.events.some((e: EventData) => {
        const eName = (e.event_name || e.name || '').toLowerCase();
        return eName.includes('payday') || eName.includes('fiesta') || eName.includes('regla');
      })
        ? 1
        : 0;

    const holiday_flag =
      inputData.events &&
      inputData.events.some((e: EventData) => {
        const eType = (e.event_type || e.type || '').toLowerCase();
        const eName = (e.event_name || e.name || '').toLowerCase();
        return eType === 'holiday' || eName.includes('holiday') || eName.includes('lent') || eName.includes('meatless') || eName.includes('holy week') || eName.includes('easter');
      })
        ? 1
        : 0;

    const nearby_event_flag = hasEvents ? 1 : 0;
    const typhoon_flag =
      inputData.events &&
      inputData.events.some((e: EventData) => {
        const eName = (e.event_name || e.name || '').toLowerCase();
        const eType = (e.event_type || e.type || '').toLowerCase();
        return eName.includes('typhoon') || eName.includes('tcws') || eName.includes('cyclone') || eType.includes('typhoon');
      })
        ? 1
        : 0;

    let startDateStr: string | null = null;
    if (inputData.events && inputData.events.length > 0) {
      const startDates = inputData.events
        .map((e: EventData) => e.start_date || e.date)
        .filter(Boolean) as string[];
      if (startDates.length > 0) {
        startDateStr = startDates.reduce(
          (min: string, d: string) => (d < min ? d : min),
          startDates[0]
        );
      }
    }

    let dateObj = new Date();
    if (startDateStr) {
      dateObj = new Date(startDateStr);
    }

    const isSchoolEvent =
      inputData.events &&
      inputData.events.some((e: EventData) => {
        const eName = (e.event_name || e.name || '').toLowerCase();
        return eName.includes('school') || eName.includes('academic');
      });
    const month = isSchoolEvent ? 8 : dateObj.getMonth() + 1;
    const jsDay = dateObj.getDay();
    const day_of_week = jsDay === 0 ? 6 : jsDay - 1;
    const weekend_flag = day_of_week === 4 || day_of_week === 5 || day_of_week === 6 ? 1 : 0;
    const storeMetadata: Record<string, { store_type: string; area_type: string }> = {
      '007113ac-1c31-4b8f-9225-d1d9950d10b5': {
        store_type: 'Convenience',
        area_type: 'Escario Central'
      },
      'ecc700f5-a37a-463d-96cc-ca0b65549811': { store_type: 'Convenience', area_type: 'IT Park' },
      '41c06a70-efd6-404a-8593-eecc5841b3ce': {
        store_type: 'Convenience',
        area_type: 'Gen Maxilom'
      },
      '77e30ef5-f46c-41cb-96fd-981bf35a069e': {
        store_type: 'Convenience',
        area_type: 'SRP-Talisay'
      },
      '0692009e-1bd9-4fd5-8610-3c7f07304d16': {
        store_type: 'Convenience',
        area_type: 'Tabunok Market'
      },
      '2649df63-6080-4797-83f6-923f114a2c9e': {
        store_type: 'Convenience',
        area_type: 'MEZ Highway'
      },
      '53f4af12-5ad2-43f7-b8b6-d00d8686fa2c': {
        store_type: 'Convenience',
        area_type: 'Marina Mall'
      }
    };
    const meta = storeMetadata[inputData.store.id] || {
      store_type: 'Convenience',
      area_type: 'IT Park'
    };

    let holiday_type = 'No Holiday';
    if (holiday_flag === 1) {
      const holidayEvent = inputData.events?.find((e: EventData) => {
        const eType = (e.event_type || e.type || '').toLowerCase();
        const eName = (e.event_name || e.name || '').toLowerCase();
        return eType === 'holiday' || eName.includes('holiday') || eName.includes('lent') || eName.includes('meatless') || eName.includes('holy week') || eName.includes('easter');
      });
      const name = (holidayEvent?.event_name || holidayEvent?.name || '').toLowerCase();
      if (name.includes('christmas') || name.includes('xmas')) {
        holiday_type = 'Christmas';
      } else if (
        name.includes('holy week') ||
        name.includes('easter') ||
        name.includes('lent') ||
        name.includes('meatless') ||
        name.includes('thursday') ||
        name.includes('friday')
      ) {
        holiday_type = 'Holy Week';
      } else {
        holiday_type = 'Holy Week'; // default holiday
      }
    }

    const hour_of_day = new Date().getHours();
    const prodDetail = inv;

    const brownout_flag =
      inv.brownout_flag ??
      (inputData.events &&
      inputData.events.some((e: EventData) => {
        const eName = (e.event_name || e.name || '').toLowerCase();
        const eType = (e.event_type || e.type || '').toLowerCase();
        return (
          eName.includes('brownout') ||
          eName.includes('power') ||
          eName.includes('grid') ||
          eType.includes('brownout') ||
          eType.includes('grid')
        );
      })
        ? 1
        : 0);

    const temp = inputData.environmental_signals?.temp ?? 32;
    const rainfall = inputData.environmental_signals?.precipitation ?? 0;
    const heatIndex = inputData.environmental_signals?.heat_index ?? 35;

    return {
      store_id: inputData.store.id,
      store_type: meta.store_type,
      area_type: meta.area_type,
      product_id: inv.sku || inv.product_id,
      category: category,
      price: prodDetail?.price ?? 50,
      inventory: inv.current_stock,
      qty_sold: Math.round(inv.base_demand / 10), // heuristic for qty_sold
      temperature: temp,
      forecasted_temperature: temp,
      rainfall: rainfall,
      forecasted_rainfall: rainfall,
      heat_index: heatIndex,
      forecasted_heat_index: heatIndex,
      foot_traffic: foot_traffic,
      foot_traffic_lag_2h: foot_traffic,
      promo_flag: promo_flag,
      payday_flag: payday_flag,
      weekend_flag: weekend_flag,
      holiday_flag: holiday_flag,
      holiday_type: holiday_type,
      nearby_event_flag: nearby_event_flag,
      typhoon_flag: typhoon_flag,
      hour_of_day: hour_of_day,
      day_of_week: day_of_week,
      month: month,
      stockout_flag: inv.stockout_flag ?? 0,
      brownout_flag: brownout_flag,
      qty_sold_lag_24h: inv.qty_sold_lag_24h ?? 0,
      store_historical_avg_sales: inv.store_historical_avg_sales ?? 0,
      product_historical_avg_sales: inv.product_historical_avg_sales ?? 0,
      rolling_mean_demand_3h: inv.rolling_mean_demand_3h ?? 0,
      rolling_mean_demand_6h: inv.rolling_mean_demand_6h ?? 0,
      rolling_mean_demand_24h: inv.rolling_mean_demand_24h ?? 0,
      rolling_std_6h: inv.rolling_std_6h ?? 0
    };
  });
}
