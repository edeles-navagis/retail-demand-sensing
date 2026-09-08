import joblib
import pandas as pd
import numpy as np

import sys
import os

# Ensure the root is in PYTHONPATH so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.demand_sensing.features import FEATURE_COLUMNS, CATEGORICAL_COLS

# =========================================================
# 1. LOAD EXPORTED FILES
# =========================================================
print("Loading models and assets...")
model_multi = joblib.load("output/demand_model_multi.pkl")
encoder_multi = joblib.load("output/categorical_encoder.pkl")

PRODUCT_DESCRIPTIONS = {
    "SKU-BAK-AL-004": "All-Purpose Cream (250ml)",
    "SKU-BEV-3--005": "3-in-1 Iced Coffee Mix (Sachet)",
    "SKU-SKI-GA-076": "Skin Care Products",
    "SKU-PER-RE-071": "Cooling Talcum Powder",
    "SKU-CAN-CE-029": "Canned Sardines (155g)",
    "SKU-CAN-CA-030": "Instant Noodles (Pouch)",
    "SKU-ELE-7--038": "D-Cell Batteries (2-Pack)",
    "SKU-HOM-MO-054": "Emergency Candles (Pack of 4)",
    "SKU-BEV-BO-006": "Soda 1.5L Bottle",
    "SKU-APP-WO-002": "Wool Sweaters",
    "SKU-BAK-CO-003": "Condensed Milk (300ml)",
    "SKU-CAN-IN-031": "Instant Pancit Canton",
    "SKU-FRO-CO-025": "Frozen Treats",
    "SKU-FRE-CE-046": "Century Eggs (2-Pack)",
    "SKU-HEA-AN-047": "Anti-Dengue Lotion (Sachet)",
    "SKU-HEA-PA-048": "Paracetamol (Biogesic)",
    "SKU-ELE-D--039": "D-Cell Batteries (2-Pack)",
    "SKU-ELE-PO-040": "Power Bank (10k mAh)",
    "SKU-HOT-PO-059": "Hot Pot / Noodles",
    "SKU-HOM-EM-055": "Emergency Candles (Pack of 4)",
    "SKU-HEA-VI-050": "Vitamin C Syrup (Pediatric)",
    "SKU-CAT-PR-000": "Pet Supplies",
    "SKU-HEA-OR-049": "Oral Rehydration Salts",
    "SKU-CAN-LU-032": "Luncheon Meat (340g)",
    "SKU-CAN-CA-033": "Canned Tuna (Flakes in Oil)",
    "SKU-FRE-GO-045": "Fresh Produce / Greens",
    "SKU-SKI-SU-077": "Sunblock Sachet (SPF 50)",
    "SKU-SNA-SK-078": "SkyFlakes Crackers (Pack)",
    "SKU-STA-IN-079": "Intermediate Pad",
    "SKU-STA-BL-080": "Black Ballpen (Single)"
}

# Helper function to create base input data
def create_base_input(product_id):
    return {
        "store_id": "007113ac-1c31-4b8f-9225-d1d9950d10b5",
        "store_type": "Convenience",
        "area_type": "Escario Central",
        "product_id": product_id,
        "category": "Beverages", # Dummy category, model treats it as categorical
        "price": 50.0,
        "inventory": 100,
        "qty_sold": 5,
        "foot_traffic": 50,
        "promo_flag": 0,
        "payday_flag": 0,
        "weekend_flag": 0,
        "holiday_flag": 0,
        "holiday_type": "No Holiday",
        "nearby_event_flag": 0,
        "stockout_flag": 0,
        "brownout_flag": 0,
        "forecasted_temperature": 30.0,
        "forecasted_rainfall": 0.0,
        "forecasted_heat_index": 30.0,
        "hour_of_day": 12,
        "day_of_week": 3,
        "month": 4,
        "qty_sold_lag_1h": 5.0,
        "qty_sold_lag_24h": 5.0,
        "foot_traffic_lag_1h": 50.0,
        "foot_traffic_lag_2h": 50.0,
        "store_historical_avg_sales": 5.0,
        "product_historical_avg_sales": 5.0,
        "rolling_mean_demand_3h": 5.0,
        "rolling_mean_demand_6h": 5.0,
        "rolling_mean_demand_24h": 5.0,
        "rolling_std_6h": 1.0
    }

def run_prediction(scenario_name, product_ids, modifications):
    print(f"\n[{scenario_name}]")
    for pid in product_ids:
        data = create_base_input(pid)
        data.update(modifications)
        df = pd.DataFrame([data])
        
        df[CATEGORICAL_COLS] = encoder_multi.transform(df[CATEGORICAL_COLS].astype(str))
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype(int)
        
        df = df[FEATURE_COLUMNS]
        
        preds = model_multi.predict(df)[0]
        pred_1h = max(0.0, float(preds[0]))
        pred_2h = max(0.0, float(preds[1]))
        pred_3h = max(0.0, float(preds[2]))
        
        desc = PRODUCT_DESCRIPTIONS.get(pid, "Unknown Product")
        print(f"Product: {pid} ({desc}) | Predicted 1h: {pred_1h:.2f} | 2h: {pred_2h:.2f} | 3h: {pred_3h:.2f}")


# =========================================================
# 2. RUN TEST CASES
# =========================================================

# Case 1: High Heat Index
run_prediction(
    "Case 1: High Heat Index",
    ["SKU-BAK-AL-004", "SKU-BEV-3--005", "SKU-SKI-GA-076", "SKU-PER-RE-071"],
    {"forecasted_heat_index": 45.0, "forecasted_temperature": 38.0}
)

# Case 2: Typhoon (Panic Buying)
run_prediction(
    "Case 2: Typhoon / Panic Buying",
    ["SKU-CAN-CE-029", "SKU-CAN-CA-030", "SKU-ELE-7--038", "SKU-HOM-MO-054"],
    {"forecasted_rainfall": 15.0}
)

# Case 3: Barangay Fiesta + Payday
run_prediction(
    "Case 3: Barangay Fiesta + Payday",
    ["SKU-BEV-BO-006", "SKU-APP-WO-002", "SKU-BAK-CO-003", "SKU-CAN-IN-031", "SKU-FRO-CO-025"],
    {"nearby_event_flag": 1, "payday_flag": 1}
)

# Case 4: Prolonged Rain
run_prediction(
    "Case 4: Prolonged Rain + Humidity",
    ["SKU-FRE-CE-046", "SKU-HEA-AN-047", "SKU-HEA-PA-048"],
    {"forecasted_rainfall": 8.0, "month": 7}
)

# Case 5: Brownout / Blackout Prep
run_prediction(
    "Case 5: Brownout Prep + High Heat",
    ["SKU-ELE-D--039", "SKU-ELE-PO-040"],
    {"brownout_flag": 1, "forecasted_heat_index": 38.0, "forecasted_temperature": 36.0}
)

# Case 6: Rush Hour
run_prediction(
    "Case 6: Rush Hour",
    ["SKU-HOT-PO-059", "SKU-HOM-EM-055", "SKU-HEA-VI-050"],
    {"hour_of_day": 18} # 6 PM
)

# Case 7: Rainy, Flu Season
run_prediction(
    "Case 7: Rainy, Flu season",
    ["SKU-CAT-PR-000", "SKU-HEA-OR-049"],
    {"month": 7, "forecasted_rainfall": 3.0} # July
)

# Case 8: Holy Week
run_prediction(
    "Case 8: Holy Week",
    ["SKU-CAN-LU-032", "SKU-CAN-CA-033", "SKU-FRE-GO-045", "SKU-SKI-SU-077"],
    {"holiday_type": "Holy Week", "holiday_flag": 1}
)

# Case 9: School Season
run_prediction(
    "Case 9: School Season",
    ["SKU-SNA-SK-078", "SKU-STA-IN-079", "SKU-STA-BL-080"],
    {"month": 8} # August
)

print("\nTesting completed.")