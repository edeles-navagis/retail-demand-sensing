import pandas as pd
import numpy as np
import os

filename = 'synthetic_cstore_sales_multi_store.csv'

if not os.path.exists(filename):
    print(f"Error: {filename} does not exist. Please run create_data_set.py first!")
    exit(1)

print(f"Loading a sample from {filename}...")
# Read first 500,000 rows to speed up validation
df = pd.read_csv(filename, nrows=500000)

print("\n--- Column Existence Check ---")
expected_columns = [
    'timestamp', 'store_id', 'area_type', 'product_id', 'product_name',
    'category', 'price', 'inventory', 'qty_sold', 'true_demand', 'hour_of_day', 'day_of_week',
    'weekend_flag', 'holiday_type', 'payday_flag', 'promo_flag',
    'forecasted_temperature', 'forecasted_rainfall', 'forecasted_heat_index',
    'foot_traffic', 'nearby_event_flag', 'stockout_flag',
    'pagasa_heatwave_flag', 'typhoon_panic_flag',
    'grid_maintenance_flag', 'back_to_school_flag',
    'qty_sold_lag_1h', 'qty_sold_lag_24h', 'qty_sold_lag_168h',
    'foot_traffic_lag_1h', 'foot_traffic_lag_2h',
    'store_historical_avg_sales', 'product_historical_avg_sales'
]

missing_cols = [c for c in expected_columns if c not in df.columns]
if missing_cols:
    print(f"FAIL: Missing columns: {missing_cols}")
else:
    print("PASS: All expected columns are present.")

removed_columns = ['store_type', 'holiday_flag', 'barangay_fiesta_flag', 'lenten_meatless_flag']
present_removed = [c for c in removed_columns if c in df.columns]
if present_removed:
    print(f"FAIL: These columns were removed as redundant duplicates but are still present: {present_removed}")
else:
    print("PASS: Redundant duplicate columns (store_type, holiday_flag, barangay_fiesta_flag, lenten_meatless_flag) are absent, as expected.")

print("\n--- target variable check (true_demand vs qty_sold) ---")
# 1. Check true_demand == qty_sold when stockout_flag == 0 and inventory > 0
normal_mask = (df['stockout_flag'] == 0) & (df['inventory'] > 0)
normal_df = df[normal_mask]
if len(normal_df) > 0:
    diff_normal = (normal_df['true_demand'] != normal_df['qty_sold']).sum()
    if diff_normal == 0:
        print("PASS: true_demand matches qty_sold during normal conditions (no stockout, inventory > 0).")
    else:
        print(f"FAIL: true_demand differs from qty_sold in {diff_normal} rows under normal conditions.")
else:
    print("WARNING: No normal condition rows found in the sample.")

# 2. Check true_demand imputation when stockout_flag == 1 or inventory == 0
stockout_mask = (df['stockout_flag'] == 1) | (df['inventory'] == 0)
stockout_df = df[stockout_mask]
if len(stockout_df) > 0:
    # On average, true_demand should be higher or different than qty_sold under stockout conditions
    imputed_diff = (stockout_df['true_demand'] != stockout_df['qty_sold']).sum()
    print(f"INFO: Under stockout/depleted inventory, true_demand was imputed differently from qty_sold in {imputed_diff} / {len(stockout_df)} rows.")
    # Check that true_demand is filled (no NaNs)
    nans = stockout_df['true_demand'].isna().sum()
    if nans == 0:
        print("PASS: No NaNs in true_demand column under stockout conditions.")
    else:
        print(f"FAIL: Found {nans} NaNs in true_demand under stockout conditions.")
else:
    print("WARNING: No stockout condition rows found in the sample.")

print("\n--- temporal shifting lag check ---")
# Pick a single store and product to verify lags
sample_store = df['store_id'].iloc[0]
sample_product = df['product_id'].iloc[0]
sub_df = df[(df['store_id'] == sample_store) & (df['product_id'] == sample_product)].sort_values('timestamp').copy()

if len(sub_df) > 200:
    # Check qty_sold_lag_1h: qty_sold shifted by 1 hour
    computed_lag_1h = sub_df['qty_sold'].shift(1)
    lag_1h_diff = (sub_df['qty_sold_lag_1h'].iloc[1:] != computed_lag_1h.iloc[1:]).sum()
    
    # Check qty_sold_lag_24h
    computed_lag_24h = sub_df['qty_sold'].shift(24)
    lag_24h_diff = (sub_df['qty_sold_lag_24h'].iloc[24:] != computed_lag_24h.iloc[24:]).sum()
    
    # Check foot_traffic_lag_1h
    computed_ft_lag_1h = sub_df['foot_traffic'].shift(1)
    ft_lag_1h_diff = (sub_df['foot_traffic_lag_1h'].iloc[1:] != computed_ft_lag_1h.iloc[1:]).sum()
    
    if lag_1h_diff == 0 and lag_24h_diff == 0 and ft_lag_1h_diff == 0:
        print("PASS: Lag features are correctly aligned and shifted.")
    else:
        print(f"FAIL: Lag discrepancies found: lag_1h_diff={lag_1h_diff}, lag_24h_diff={lag_24h_diff}, ft_lag_1h_diff={ft_lag_1h_diff}")
else:
    print("WARNING: Insufficient sample size for lag validation.")

print("\n--- Weather column check ---")
weather_cols = ['forecasted_temperature', 'forecasted_rainfall', 'forecasted_heat_index']
for col in weather_cols:
    if col in df.columns:
        null_count = df[col].isna().sum()
        mean_val = df[col].mean()
        print(f"INFO: {col} has mean={mean_val:.2f}, nulls={null_count} ({null_count/len(df)*100:.2f}%)")
    else:
        print(f"FAIL: weather column {col} missing.")

print("\n--- Named scenario flag check ---")
scenario_flags = ['pagasa_heatwave_flag', 'typhoon_panic_flag', 'grid_maintenance_flag', 'back_to_school_flag']
for flag in scenario_flags:
    if flag in df.columns:
        vc = df[flag].value_counts().to_dict()
        print(f"INFO: {flag} distribution: {vc}")
        if set(df[flag].unique()) - {0, 1}:
            print(f"FAIL: {flag} contains values outside {{0,1}}.")
    else:
        print(f"FAIL: expected scenario flag column {flag} missing.")

print("\n--- High-cardinality target encoding check ---")
# Verify store and product baseline features are present and non-empty
se_nans = df['store_historical_avg_sales'].isna().sum()
pe_nans = df['product_historical_avg_sales'].isna().sum()

if se_nans == 0 and pe_nans == 0:
    print("PASS: Global baseline averages are populated for all rows.")
    # Check that they represent historical averages
    first_store_avg = df[df['store_id'] == sample_store]['store_historical_avg_sales'].iloc[0]
    expected_store_avg = df[df['store_id'] == sample_store]['qty_sold'].mean()
    # Note: df is a subset (first 500k rows), while target encoding is computed globally.
    # So we print them to check reasonableness.
    print(f"INFO: Store baseline sales for {sample_store}: {first_store_avg} (Sample subset mean: {expected_store_avg:.4f})")
else:
    print(f"FAIL: Target encoding columns contain NaNs. store_nans={se_nans}, product_nans={pe_nans}")

print("\nValidation complete.")
