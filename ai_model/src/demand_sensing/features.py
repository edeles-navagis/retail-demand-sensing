import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

UNKNOWN_CATEGORY_FLAG = 999999

CATEGORICAL_COLS = [
    "store_id", "area_type", "product_id", "category", "holiday_type"
]

FEATURE_COLUMNS = [
    "store_id", "area_type", "product_id", "category",
    "price", "inventory", 
    "promo_flag", "payday_flag", "weekend_flag", "holiday_type", "nearby_event_flag", "stockout_flag", "brownout_flag",
    "forecasted_temperature", "forecasted_rainfall", "forecasted_heat_index", 
    "hour_of_day", "day_of_week", "month",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow",
    "is_rush_hour", "is_heatwave_risk", "is_heavy_rain", "is_fiesta_payday", "is_school_season",
    "qty_sold_lag_1h", "qty_sold_lag_24h",
    "foot_traffic_lag_1h", "foot_traffic_lag_2h",
    "store_historical_avg_sales", "product_historical_avg_sales",
    "rolling_mean_demand_3h", "rolling_mean_demand_6h", "rolling_mean_demand_24h", "rolling_std_6h"
]

def load_data(filepath: str) -> pd.DataFrame:
    print("Loading data...")
    df = pd.read_csv(filepath, dtype={'holiday_type': str, 'area_type': str})
    df["holiday_type"] = df["holiday_type"].fillna("No Holiday")
    if "area_type" in df.columns:
        df["area_type"] = df["area_type"].fillna("Convenience")
    else:
        df["area_type"] = "Escario Central"
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def get_noise(shape, mean=0, std=5) -> np.ndarray:
    return np.random.normal(mean, std, size=shape).round().astype(int)

def inject_stochastic_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    print("Injecting realistic demand logic for specific scenarios...")
    np.random.seed(42)
    
    mask_heat = df['forecasted_heat_index'] >= 40
    heat_products = ["SKU-BAK-AL-004", "SKU-BEV-3--005", "SKU-SKI-GA-076", "SKU-PER-RE-071"]
    heat_mask = mask_heat & df['product_id'].isin(heat_products)
    df.loc[heat_mask, 'true_demand'] += (50 + get_noise(df[heat_mask].shape[0]))

    mask_typhoon = df['forecasted_rainfall'] >= 10
    typhoon_products = ["SKU-CAN-CE-029", "SKU-CAN-CA-030", "SKU-ELE-7--038", "SKU-HOM-MO-054"]
    typhoon_mask = mask_typhoon & df['product_id'].isin(typhoon_products)
    df.loc[typhoon_mask, 'true_demand'] += (80 + get_noise(df[typhoon_mask].shape[0], std=8))

    mask_fiesta_payday = (df['nearby_event_flag'] == 1) & (df['payday_flag'] == 1)
    fiesta_products = ["SKU-BEV-BO-006", "SKU-APP-WO-002", "SKU-BAK-CO-003", "SKU-CAN-IN-031", "SKU-FRO-CO-025"]
    fiesta_mask = mask_fiesta_payday & df['product_id'].isin(fiesta_products)
    df.loc[fiesta_mask, 'true_demand'] += (60 + get_noise(df[fiesta_mask].shape[0]))

    mask_rain = (df['forecasted_rainfall'] >= 5)
    rain_products = ["SKU-FRE-CE-046", "SKU-HEA-AN-047", "SKU-HEA-PA-048"]
    rain_mask = mask_rain & df['product_id'].isin(rain_products)
    df.loc[rain_mask, 'true_demand'] += (40 + get_noise(df[rain_mask].shape[0]))

    if 'brownout_flag' not in df.columns:
        if 'grid_maintenance_flag' in df.columns:
            df['brownout_flag'] = df['grid_maintenance_flag']
        else:
            df['brownout_flag'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
    mask_brownout = (df['brownout_flag'] == 1) & (df['forecasted_heat_index'] >= 35)
    brownout_products = ["SKU-ELE-D--039", "SKU-ELE-PO-040"]
    brownout_mask = mask_brownout & df['product_id'].isin(brownout_products)
    df.loc[brownout_mask, 'true_demand'] += (45 + get_noise(df[brownout_mask].shape[0]))

    df["hour_of_day"] = df["timestamp"].dt.hour
    mask_rush = df['hour_of_day'].isin([7,8,9,17,18,19])
    rush_products = ["SKU-HOT-PO-059", "SKU-HOM-EM-055", "SKU-HEA-VI-050"]
    rush_mask = mask_rush & df['product_id'].isin(rush_products)
    df.loc[rush_mask, 'true_demand'] += (30 + get_noise(df[rush_mask].shape[0], std=3))

    df["month"] = df["timestamp"].dt.month
    mask_flu = df['month'].isin([6,7,8]) & (df['forecasted_rainfall'] > 0)
    flu_products = ["SKU-CAT-PR-000", "SKU-HEA-OR-049"]
    flu_mask = mask_flu & df['product_id'].isin(flu_products)
    df.loc[flu_mask, 'true_demand'] += (35 + get_noise(df[flu_mask].shape[0], std=4))

    mask_holy = df['holiday_type'] == 'Holy Week'
    holy_products = ["SKU-CAN-LU-032", "SKU-CAN-CA-033", "SKU-FRE-GO-045", "SKU-SKI-SU-077"]
    holy_mask = mask_holy & df['product_id'].isin(holy_products)
    df.loc[holy_mask, 'true_demand'] += (50 + get_noise(df[holy_mask].shape[0]))

    mask_school = df['month'].isin([8, 9])
    school_products = ["SKU-SNA-SK-078", "SKU-STA-IN-079", "SKU-STA-BL-080"]
    school_mask = mask_school & df['product_id'].isin(school_products)
    df.loc[school_mask, 'true_demand'] += (60 + get_noise(df[school_mask].shape[0]))

    df['true_demand'] = np.maximum(df['true_demand'], 0)

    # Sort strictly for sequential window calculations: Store -> Product -> Time
    df = df.sort_values(by=["store_id", "product_id", "timestamp"]).reset_index(drop=True)
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    return df

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    if "area_type" not in df.columns:
        df["area_type"] = "Escario Central"
    df["area_type"] = df["area_type"].fillna("Escario Central")

    if "hour_of_day" in df.columns:
        if "sin_hour" not in df.columns:
            df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0)
            df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0)
        if "is_rush_hour" not in df.columns:
            df["is_rush_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    if "day_of_week" in df.columns:
        if "sin_dow" not in df.columns:
            df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
            df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    if "forecasted_heat_index" in df.columns and "is_heatwave_risk" not in df.columns:
        df["is_heatwave_risk"] = (df["forecasted_heat_index"] >= 40).astype(int)

    if "forecasted_rainfall" in df.columns and "is_heavy_rain" not in df.columns:
        df["is_heavy_rain"] = (df["forecasted_rainfall"] >= 10).astype(int)

    if "is_fiesta_payday" not in df.columns:
        nearby = df.get("nearby_event_flag", pd.Series(0, index=df.index))
        payday = df.get("payday_flag", pd.Series(0, index=df.index))
        df["is_fiesta_payday"] = ((nearby == 1) & (payday == 1)).astype(int)

    if "month" in df.columns and "is_school_season" not in df.columns:
        df["is_school_season"] = df["month"].isin([8, 9]).astype(int)

    if "foot_traffic_lag_1h" not in df.columns:
        df["foot_traffic_lag_1h"] = df.get("foot_traffic_lag_2h", df.get("foot_traffic", pd.Series(50.0, index=df.index)))

    if "qty_sold_lag_1h" not in df.columns:
        df["qty_sold_lag_1h"] = df.get("qty_sold_lag_24h", df.get("qty_sold", pd.Series(0.0, index=df.index)))

    if "holiday_type" not in df.columns:
        df["holiday_type"] = "No Holiday"
    df["holiday_type"] = df["holiday_type"].fillna("No Holiday")

    return df

def generate_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating target and generating features...")
    df["target_demand_1h"] = df.groupby(["store_id", "product_id"], observed=False)["true_demand"].shift(-1)
    df["target_demand_2h"] = df.groupby(["store_id", "product_id"], observed=False)["true_demand"].shift(-2)
    df["target_demand_3h"] = df.groupby(["store_id", "product_id"], observed=False)["true_demand"].shift(-3)

    grouped = df.groupby(["store_id", "product_id"], observed=False)

    # Rolling windows shifted by 2h to safely serve t+2 predictions without horizon leakage
    df["rolling_mean_demand_3h"] = grouped["qty_sold"].transform(lambda x: x.shift(2).rolling(window=3).mean())
    df["rolling_mean_demand_6h"] = grouped["qty_sold"].transform(lambda x: x.shift(2).rolling(window=6).mean())
    df["rolling_mean_demand_24h"] = grouped["qty_sold"].transform(lambda x: x.shift(2).rolling(window=24).mean())
    df["rolling_std_6h"] = grouped["qty_sold"].transform(lambda x: x.shift(2).rolling(window=6).std())

    # Expanding historical averages resolved chronologically:
    # 1. store-level baseline sales velocity
    df["store_historical_avg_sales"] = df.groupby("store_id")["qty_sold"].transform(lambda x: x.shift(2).expanding().mean()).fillna(0)
    # 2. product-level cross-store baseline velocity
    df["product_historical_avg_sales"] = df.groupby("product_id")["qty_sold"].transform(lambda x: x.shift(2).expanding().mean()).fillna(0)

    # Compute derived scenario and cyclical features
    df = compute_derived_features(df)

    if "qty_sold_lag_1h" not in df.columns:
        df["qty_sold_lag_1h"] = grouped["qty_sold"].shift(1).fillna(0)
    else:
        df["qty_sold_lag_1h"] = df["qty_sold_lag_1h"].fillna(0)

    if "foot_traffic_lag_1h" not in df.columns:
        df["foot_traffic_lag_1h"] = grouped["foot_traffic"].shift(1).fillna(50.0)
    else:
        df["foot_traffic_lag_1h"] = df["foot_traffic_lag_1h"].fillna(50.0)

    # Drop rows where target or rolling features are NaN due to shifting
    df = df.dropna(subset=["target_demand_1h", "target_demand_2h", "target_demand_3h", "rolling_mean_demand_24h"])

    if len(df) == 0:
        raise ValueError("Dataset empty after lag filtering. Verify time coverage limits.")
        
    return df

def split_and_encode_data(df: pd.DataFrame):
    print("Splitting datasets chronologically...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    unique_dates = sorted(df["timestamp"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]

    train_df = df[df["timestamp"] < split_date].copy()
    test_df = df[df["timestamp"] >= split_date].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"Splitting failed! Train rows: {len(train_df)}, Test rows: {len(test_df)}.")

    print("Fitting Ordinal Encoder to intercept cold-start unseen inference categories...")
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value', 
        unknown_value=UNKNOWN_CATEGORY_FLAG
    )

    train_df[CATEGORICAL_COLS] = encoder.fit_transform(train_df[CATEGORICAL_COLS].astype(str))
    test_df[CATEGORICAL_COLS] = encoder.transform(test_df[CATEGORICAL_COLS].astype(str))

    for col in CATEGORICAL_COLS:
        train_df[col] = train_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    y_train_multi = train_df[["target_demand_1h", "target_demand_2h", "target_demand_3h"]]
    y_test_multi = test_df[["target_demand_1h", "target_demand_2h", "target_demand_3h"]]

    print(f"Training set matrices initialized: {len(X_train)} rows.")
    print(f"Testing set matrices initialized: {len(X_test)} rows.")

    return X_train, X_test, y_train_multi, y_test_multi, encoder

def prepare_pipeline_data(filepath: str):
    df = load_data(filepath)
    df = inject_stochastic_scenarios(df)
    df = generate_causal_features(df)
    return split_and_encode_data(df)
