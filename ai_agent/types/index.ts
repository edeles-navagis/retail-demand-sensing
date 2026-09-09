export interface StoreContext {
  id: string;
  name?: string;
  classification?: string;
  latitude?: number;
  longitude?: number;
  geographical_profile?: string;
}

export interface EnvironmentalSignals {
  temp?: number;
  heat_index?: number;
  uv_index?: number;
  precipitation?: number;
}

export interface EventData {
  name?: string;
  event_name?: string;
  type?: string;
  event_type?: string;
  date?: string;
  start_date?: string;
  crowd?: string | number;
  expected_crowd?: string | number;
}

export interface InventoryItem {
  product_id: string;
  sku?: string;
  name?: string;
  category?: string;
  sensitivity?: string;
  price?: number;
  wholesale_price?: number;
  wholesale_bundle_qty?: number;
  wholesale_gross_weight_kg?: number;
  wholesale_gross_volume_m3?: number;
  current_stock: number;
  base_demand: number;
  predicted_demand_1h?: number;
  predicted_demand_2h?: number;
  predicted_demand_3h?: number;
  historical_sales_t_minus_1?: number;
  historical_sales_t_minus_2?: number;
  historical_sales_t_minus_3?: number;
  stockout_flag?: number;
  brownout_flag?: number;
  qty_sold_lag_24h?: number;
  store_historical_avg_sales?: number;
  product_historical_avg_sales?: number;
  rolling_mean_demand_3h?: number;
  rolling_mean_demand_6h?: number;
  rolling_mean_demand_24h?: number;
  rolling_std_6h?: number;
}

export interface ProductDna {
  id?: string;
  product_id?: string;
  name?: string;
  category?: string;
  sensitivity?: string;
  price?: number;
  wholesale_bundle_qty?: number;
}

export interface Warehouse {
  id: string;
  name?: string;
  travel_time_mins?: number;
  truck_weight_limit?: number;
  truck_volume_limit?: number;
  truck_flat_rate?: number;
  truck_rate?: number;
  distance_km?: number;
}

export interface AiDemandSensingSettings {
  safe_demand_threshold_pct: number;
  budget_limit_php: number;
  product_buffer_factors?: { category: string; buffer_factor: number }[];
  critical_hours: number;
  mid_hours: number;
}

export interface AgentInputData {
  store: StoreContext;
  environmental_signals?: EnvironmentalSignals;
  events?: EventData[];
  inventory?: InventoryItem[];
  product_dna?: ProductDna[];
  warehouses?: Warehouse[];
  settings?: AiDemandSensingSettings;
  computed_recommendations?: any[];
}

export interface PredictionResult {
  product_id: string;
  impact_factor: 'CRITICAL' | 'MID' | 'LOW' | 'SAFE' | string;
  projected_demand_increase_pct: number;
  recommended_replenishment: number;
  source_warehouse_id: string;
  estimated_delivery_arrival_time: number;
}

export interface CriticalInventory {
  product_id: string;
  current_quantity: number;
  expected_quantity: number;
  predicted_demand_1h?: number;
  predicted_demand_2h?: number;
  predicted_demand_3h?: number;
}

export interface AgentResponse {
  store_id: string;
  confidence_score: number;
  reasoning_log: string;
  predictions: PredictionResult[];
  critical_product_inventory: CriticalInventory[];
  approval_status?: 'AUTO_APPROVED' | 'MANUAL_APPROVAL';
}

export interface ModelInstance {
  store_id: string;
  product_id: string;
  category: string;
  price: number;
  inventory: number;
  promo_flag: number;
  payday_flag: number;
  weekend_flag: number;
  holiday_type: string;
  nearby_event_flag: number;
  stockout_flag?: number;
  brownout_flag?: number;
  temperature?: number;
  forecasted_temperature?: number;
  rainfall?: number;
  forecasted_rainfall?: number;
  heat_index?: number;
  forecasted_heat_index?: number;
  hour_of_day: number;
  day_of_week: number;
  month: number;
  qty_sold_lag_24h?: number;
  foot_traffic?: number;
  foot_traffic_lag_2h?: number;
  store_historical_avg_sales?: number;
  product_historical_avg_sales?: number;
  rolling_mean_demand_3h?: number;
  rolling_mean_demand_6h?: number;
  rolling_mean_demand_24h?: number;
  rolling_std_6h?: number;
  // Legacy / optional attributes for backwards compatibility
  store_type?: string;
  area_type?: string;
  holiday_flag?: number;
  qty_sold?: number;
  qty_sold_lag_1h?: number;
  typhoon_flag?: number;
}
