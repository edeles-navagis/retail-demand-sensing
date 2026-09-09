import { z } from 'zod';

export const StoreContextSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  classification: z.string().optional(),
  geographical_profile: z.string().optional()
});

export const EnvironmentalSignalsSchema = z.object({
  temp: z.coerce.number().optional(),
  heat_index: z.coerce.number().optional(),
  uv_index: z.coerce.number().optional(),
  precipitation: z.coerce.number().optional()
});

export const EventDataSchema = z.object({
  event_name: z.string().optional(),
  event_type: z.string().optional(),
  start_date: z.string().optional(),
  expected_crowd: z.union([z.string(), z.number()]).optional()
});

export const InventoryItemSchema = z.object({
  product_id: z.string(),
  sku: z.string().optional(),
  name: z.string().optional(),
  category: z.string().optional(),
  sensitivity: z.string().optional(),
  price: z.coerce.number().optional(),
  wholesale_price: z.coerce.number().optional(),
  wholesale_bundle_qty: z.coerce.number().optional(),
  wholesale_gross_weight_kg: z.coerce.number().optional(),
  wholesale_gross_volume_m3: z.coerce.number().optional(),
  current_stock: z.coerce.number(),
  base_demand: z.coerce.number(),
  predicted_demand_1h: z.coerce.number().optional(),
  predicted_demand_2h: z.coerce.number().optional(),
  predicted_demand_3h: z.coerce.number().optional(),
  historical_sales_t_minus_1: z.coerce.number().optional(),
  historical_sales_t_minus_2: z.coerce.number().optional(),
  historical_sales_t_minus_3: z.coerce.number().optional(),
  stockout_flag: z.coerce.number().optional(),
  brownout_flag: z.coerce.number().optional(),
  qty_sold_lag_24h: z.coerce.number().optional(),
  store_historical_avg_sales: z.coerce.number().optional(),
  product_historical_avg_sales: z.coerce.number().optional(),
  rolling_mean_demand_3h: z.coerce.number().optional(),
  rolling_mean_demand_6h: z.coerce.number().optional(),
  rolling_mean_demand_24h: z.coerce.number().optional(),
  rolling_std_6h: z.coerce.number().optional()
});

export const WarehouseSchema = z.object({
  id: z.string(),
  name: z.string().optional(),
  travel_time_mins: z.coerce.number().optional(),
  truck_weight_limit: z.coerce.number().optional(),
  truck_volume_limit: z.coerce.number().optional(),
  truck_flat_rate: z.coerce.number().optional(),
  delivery_rate: z.coerce.number().optional(),
  truck_rate: z.coerce.number().optional(),
  distance_km: z.coerce.number().optional()
});

export const AgentInputDataSchema = z.object({
  store: StoreContextSchema,
  environmental_signals: EnvironmentalSignalsSchema.optional(),
  events: z.array(EventDataSchema).optional(),
  inventory: z.array(InventoryItemSchema).optional(),
  warehouses: z.array(WarehouseSchema).optional()
});
