import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# =========================================================
# 1. STORE PROFILES & BEHAVIOR MAPPING
# =========================================================
store_profiles = {
    '007113ac-1c31-4b8f-9225-d1d9950d10b5': {
        'name': 'Escario Central',
        'type': 'Convenience',
        'behavior': 'office_hotel'
    },
    'ecc700f5-a37a-463d-96cc-ca0b65549811': {
        'name': 'IT Park',
        'type': 'Convenience',
        'behavior': 'bpo_graveyard'
    },
    '41c06a70-efd6-404a-8593-eecc5841b3ce': {
        'name': 'Gen Maxilom',
        'type': 'Convenience',
        'behavior': 'student_nightlife'
    },
    '77e30ef5-f46c-41cb-96fd-981bf35a069e': {
        'name': 'SRP-Talisay',
        'type': 'Convenience',
        'behavior': 'commuter_weekend'
    },
    '0692009e-1bd9-4fd5-8610-3c7f07304d16': {
        'name': 'Tabunok Market',
        'type': 'Convenience',
        'behavior': 'market_mass'
    },
    '2649df63-6080-4797-83f6-923f114a2c9e': {
        'name': 'MEZ Highway',
        'type': 'Convenience',
        'behavior': 'factory_shift'
    },
    '53f4af12-5ad2-43f7-b8b6-d00d8686fa2c': {
        'name': 'Marina Mall',
        'type': 'Convenience',
        'behavior': 'airport_premium'
    }
}

# =========================================================
# 2. PRODUCT CATALOG DEFINITION (80 SKUs)
# =========================================================
# (Condensed for brevity - use your full 80 item list here)
product_data = [
    ( "SKU-CAT-PR-000", "Apparel", "Sachet Umbrella", 99 ),
    ( "SKU-APP-SA-001", "Apparel", "Wool Sweaters", 699 ),
    ( "SKU-APP-WO-002", "Baking", "Condensed Milk (300ml)", 62 ),
    ( "SKU-BAK-CO-003", "Baking", "All-Purpose Cream (250ml)", 69 ),
    ( "SKU-BAK-AL-004", "Beverages", "3-in-1 Iced Coffee Mix (Sachet)", 10 ),
    ( "SKU-BEV-3--005", "Beverages", "Bottled Distilled Water (6L)", 79 ),
    ( "SKU-BEV-BO-006", "Beverages", "Soda 1.5L Bottle", 89 ),
    ( "SKU-BEV-SO-007", "Beverages", "Slurpee (Cherry Flavor)", 39 ),
    ( "SKU-BEV-SL-008", "Beverages", "Big Gulp Coca-Cola", 49 ),
    ( "SKU-BEV-BI-009", "Beverages", "Gatorade Cool Blue", 65 ),
    ( "SKU-BEV-GA-010", "Beverages", "Brewed Coffee (12oz)", 45 ),
    ( "SKU-BEV-BR-011", "Beverages", "Iced Coffee (16oz)", 59 ),
    ( "SKU-BEV-IC-012", "Beverages", "Premium Latte (12oz)", 79 ),
    ( "SKU-BEV-PR-013", "Beverages", "Instant Coffee Sachet", 10 ),
    ( "SKU-BEV-IN-014", "Beverages", "Energy Coffee Drink", 49 ),
    ( "SKU-BEV-EN-015", "Alcoholic Beverages", "San Miguel Pale Pilsen Can", 65 ),
    ( "SKU-ALC-SA-016", "Alcoholic Beverages", "San Miguel Light Can", 69 ),
    ( "SKU-ALC-SA-017", "Alcoholic Beverages", "Red Horse Beer Can", 75 ),
    ( "SKU-ALC-RE-018", "Alcoholic Beverages", "Heineken Can", 99 ),
    ( "SKU-ALC-HE-019", "Alcoholic Beverages", "Corona Extra Bottle", 119 ),
    ( "SKU-ALC-CO-020", "Frozen Desserts", "Vanilla Ice Cream Cup", 35 ),
    ( "SKU-FRO-VA-021", "Frozen Desserts", "Chocolate Ice Cream Cup", 35 ),
    ( "SKU-FRO-CH-022", "Frozen Desserts", "Cookies & Cream Ice Cream Pint", 169 ),
    ( "SKU-FRO-CO-023", "Frozen Desserts", "Magnum Classic Ice Cream Bar", 75 ),
    ( "SKU-FRO-MA-024", "Frozen Desserts", "Cornetto Chocolate Cone", 45 ),
    ( "SKU-FRO-CO-025", "Canned Goods", "Spam Less Sodium 12oz", 259 ),
    ( "SKU-CAN-SP-026", "Canned Goods", "Campbell's Chunky Clam Chowder", 169 ),
    ( "SKU-CAN-CA-027", "Canned Goods", "Heinz Baked Beanz", 95 ),
    ( "SKU-CAN-HE-028", "Canned Goods", "Century Tuna Flakes in Oil", 45 ),
    ( "SKU-CAN-CE-029", "Canned Goods", "Canned Sardines (155g)", 28 ),
    ( "SKU-CAN-CA-030", "Canned Goods", "Instant Noodles (Pouch)", 18 ),
    ( "SKU-CAN-IN-031", "Canned Goods", "Luncheon Meat (340g)", 139 ),
    ( "SKU-CAN-LU-032", "Canned Goods", "Canned Tuna (Flakes in Oil)", 49 ),
    ( "SKU-CAN-CA-033", "Canned Goods", "Instant Pancit Canton (Calamansi)", 20 ),
    ( "SKU-CAN-IN-034", "Electronics", "Anker PowerCore 10K Power Bank", 1495 ),
    ( "SKU-ELE-AN-035", "Electronics", "Apple EarPods with Lightning Connector", 1290 ),
    ( "SKU-ELE-AP-036", "Electronics", "SanDisk Ultra 64GB MicroSDXC Card", 499 ),
    ( "SKU-ELE-SA-037", "Electronics", "7-Select Universal Wall Adapter", 189 ),
    ( "SKU-ELE-7--038", "Electronics", "D-Cell Batteries (2-Pack)", 129 ),
    ( "SKU-ELE-D--039", "Electronics", "Power Bank (10k mAh)", 899 ),
    ( "SKU-ELE-PO-040", "Electronics", "Rechargeable Emergency Fan", 799 ),
    ( "SKU-ELE-RE-041", "Fresh Grab-and-Go", "Tuna Salad Sandwich", 89 ),
    ( "SKU-FRE-TU-042", "Fresh Grab-and-Go", "Chicken Caesar Wrap", 119 ),
    ( "SKU-FRE-CH-043", "Fresh Grab-and-Go", "7-Fresh Fruit Cup", 69 ),
    ( "SKU-FRE-7--044", "Fresh Grab-and-Go", "Gourmet Chicken Karaage Onigiri", 79 ),
    ( "SKU-FRE-GO-045", "Fresh/Preserved", "Century Eggs (2-Pack)", 39 ),
    ( "SKU-FRE-CE-046", "Health Care", "Anti-Dengue Lotion (Sachet)", 20 ),
    ( "SKU-HEA-AN-047", "Health Care", "Paracetamol (Biogesic)", 8 ),
    ( "SKU-HEA-PA-048", "Health Care", "Oral Rehydration Salts", 18 ),
    ( "SKU-HEA-OR-049", "Health Care", "Vitamin C Syrup (Pediatric)", 129 ),
    ( "SKU-HEA-VI-050", "Home & Auto Essentials", "7-Select Bathroom Tissue", 35 ),
    ( "SKU-HOM-7--051", "Home & Auto Essentials", "AA Energizer Max Batteries", 199 ),
    ( "SKU-HOM-AA-052", "Home & Auto Essentials", "Type-C USB Charging Cable", 119 ),
    ( "SKU-HOM-TY-053", "Home & Auto Essentials", "Mobil 1 10W-40 Motor Oil", 599 ),
    ( "SKU-HOM-MO-054", "Home Essentials", "Emergency Candles (Pack of 4)", 49 ),
    ( "SKU-HOM-EM-055", "Hot Food", "Big Bite Classic Hotdog", 55 ),
    ( "SKU-HOT-BI-056", "Hot Food", "Crunch Time Fried Chicken", 109 ),
    ( "SKU-HOT-CR-057", "Hot Food", "Pepperoni Pizza Slice", 79 ),
    ( "SKU-HOT-PE-058", "Hot Food", "Pork Siomao", 39 ),
    ( "SKU-HOT-PO-059", "Over-the-Counter Health", "Tylenol Extra Strength Caplets", 12 ),
    ( "SKU-OVE-TY-060", "Over-the-Counter Health", "Gaviscon Double Action Chewable Tablets", 18 ),
    ( "SKU-OVE-GA-061", "Over-the-Counter Health", "Pepto-Bismol Liquid Suspension", 179 ),
    ( "SKU-OVE-PE-062", "Over-the-Counter Health", "Advil Ibuprofen 200mg Capsules", 15 ),
    ( "SKU-OVE-AD-063", "Packaged Snacks & Confectionery", "7-Select Kettle Chips", 59 ),
    ( "SKU-PAC-7--064", "Packaged Snacks & Confectionery", "Lay's Sour Cream & Onion", 69 ),
    ( "SKU-PAC-LA-065", "Packaged Snacks & Confectionery", "Snickers Chocolate Bar", 45 ),
    ( "SKU-PAC-SN-066", "Packaged Snacks & Confectionery", "Haribo Goldbears Gummies", 99 ),
    ( "SKU-PAC-HA-067", "Personal Care", "Colgate Optic White Travel Toothpaste", 89 ),
    ( "SKU-PER-CO-068", "Personal Care", "Biore UV Sunscreen Liquid", 329 ),
    ( "SKU-PER-BI-069", "Personal Care", "Gillette Blue II Disposable Razor", 49 ),
    ( "SKU-PER-GI-070", "Personal Care", "Rexona Men Ice Cool Deodorant", 149 ),
    ( "SKU-PER-RE-071", "Personal Care", "Cooling Talcum Powder", 45 ),
    ( "SKU-PER-CO-072", "Skin Care", "Cetaphil Gentle Skin Cleanser 125ml", 429 ),
    ( "SKU-SKI-CE-073", "Skin Care", "Nivea Creme Body Moisturizer", 149 ),
    ( "SKU-SKI-NI-074", "Skin Care", "Vaseline Lip Therapy Original", 79 ),
    ( "SKU-SKI-VA-075", "Skin Care", "Garnier Micellar Cleansing Water Travel Size", 149 ),
    ( "SKU-SKI-GA-076", "Skin Care", "Sunblock Sachet (SPF 50)", 29 ),
    ( "SKU-SKI-SU-077", "Snacks", "SkyFlakes Crackers (Pack)", 18 ),
    ( "SKU-SNA-SK-078", "Stationery", "Intermediate Pad (Sachet)", 8 ),
    ( "SKU-STA-IN-079", "Stationery", "Black Ballpen (Single)", 12 ),
    ( "SKU-STA-BL-080", "Stationery", "Plastic Envelope (Long)", 15)
]
products_df = pd.DataFrame(product_data, columns=["product_id", "category", "product_name", "price"])

# =========================================================
# 3. GLOBAL SIMULATION SETUP (Time & Weather)
# =========================================================
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31, 23, 59, 59)
date_range = pd.date_range(start=start_date, end=end_date, freq='h')

np.random.seed(42)
typhoon_days = [datetime(2023, 8, 12).date(), datetime(2023, 9, 2).date()]
holy_week = [datetime(2023, 4, 2).date() + timedelta(days=x) for x in range(7)]
grid_warning_days = [datetime(2023, 5, 14).date(), datetime(2023, 5, 22).date()]

ph_holidays = {
    (1, 1): "New Year's Day",
    (2, 17): "Chinese New Year",
    (3, 20): "Eid'l Fitr (Feast of Ramadhan)",
    (4, 2): "Maundy Thursday",
    (4, 3): "Good Friday",
    (4, 4): "Black Saturday",
    (4, 9): "Day of Valor (Araw ng Kagitingan)",
    (5, 1): "Labor Day",
    (5, 27): "Eid'l Adha (Feast of Sacrifice)",
    (6, 12): "Independence Day",
    (8, 21): "Ninoy Aquino Day",
    (8, 31): "National Heroes Day",
    (11, 1): "All Saints' Day",
    (11, 2): "All Souls' Day",
    (11, 30): "Bonifacio Day",
    (12, 8): "Feast of the Immaculate Conception of Mary",
    (12, 24): "Christmas Eve",
    (12, 25): "Christmas Day",
    (12, 30): "Rizal Day",
    (12, 31): "Last Day of the Year"
}

store_specific_holidays = {
    '007113ac-1c31-4b8f-9225-d1d9950d10b5': [
        ('Cebu City Charter Day', (2, 24), (2, 24)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Osmeña Day', (9, 9), (9, 9)),
        ('Serging Osmeña Jr. Day', (12, 4), (12, 4))
    ],
    'ecc700f5-a37a-463d-96cc-ca0b65549811': [
        ('Cebu City Charter Day', (2, 24), (2, 24)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Osmeña Day', (9, 9), (9, 9)),
        ('Serging Osmeña Jr. Day', (12, 4), (12, 4))
    ],
    '41c06a70-efd6-404a-8593-eecc5841b3ce': [
        ('Cebu City Charter Day', (2, 24), (2, 24)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Osmeña Day', (9, 9), (9, 9)),
        ('Serging Osmeña Jr. Day', (12, 4), (12, 4))
    ],
    '77e30ef5-f46c-41cb-96fd-981bf35a069e': [
        ('Talisay City Charter Day', (1, 12), (1, 12)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Feast of St. Teresa of Avila (Halad Inasal Festival)', (10, 15), (10, 15))
    ],
    '0692009e-1bd9-4fd5-8610-3c7f07304d16': [
        ('Talisay City Charter Day', (1, 12), (1, 12)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Feast of St. Teresa of Avila (Halad Inasal Festival)', (10, 15), (10, 15))
    ],
    '2649df63-6080-4797-83f6-923f114a2c9e': [
        ('Lapulapu Day (Battle of Mactan)', (4, 27), (4, 27)),
        ('Lapu-Lapu City Charter Day', (6, 17), (6, 17)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Nuestra Señora de Regla Patronal Fiesta', (11, 12), (11, 21))
    ],
    '53f4af12-5ad2-43f7-b8b6-d00d8686fa2c': [
        ('Lapulapu Day (Battle of Mactan)', (4, 27), (4, 27)),
        ('Lapu-Lapu City Charter Day', (6, 17), (6, 17)),
        ('Cebu Provincial Charter Day', (8, 6), (8, 6)),
        ('Nuestra Señora de Regla Patronal Fiesta', (11, 12), (11, 21))
    ]
}

filename = 'synthetic_cstore_sales_multi_store.csv'
if os.path.exists(filename):
    os.remove(filename) # Start fresh

print("Starting 4.9M row generation. Processing store by store...")

# =========================================================
# 4. STORE-BY-STORE GENERATION LOOP
# =========================================================
store_columns = [
    'timestamp', 'store_id', 'store_type', 'area_type', 'product_id', 'product_name', 
    'category', 'price', 'inventory', 'qty_sold', 'hour_of_day', 'day_of_week', 
    'weekend_flag', 'holiday_flag', 'holiday_type', 'payday_flag', 'promo_flag', 
    'forecasted_temperature', 'forecasted_rainfall', 'forecasted_heat_index', 
    'foot_traffic', 'nearby_event_flag', 'stockout_flag'
]

final_columns = [
    'timestamp', 'store_id', 'store_type', 'area_type', 'product_id', 'product_name', 
    'category', 'price', 'inventory', 'qty_sold', 'true_demand', 'hour_of_day', 'day_of_week', 
    'weekend_flag', 'holiday_flag', 'holiday_type', 'payday_flag', 'promo_flag', 
    'forecasted_temperature', 'forecasted_rainfall', 'forecasted_heat_index', 
    'foot_traffic', 'nearby_event_flag', 'stockout_flag',
    'qty_sold_lag_1h', 'qty_sold_lag_24h', 'qty_sold_lag_168h',
    'foot_traffic_lag_1h', 'foot_traffic_lag_2h'
]

# Write headers once
pd.DataFrame(columns=final_columns).to_csv(filename, index=False)

for store_id, profile in store_profiles.items():
    print(f"-> Generating data for {profile['name']} ({profile['behavior']})...")
    data_rows = []
    
    # Independent inventory tracking per store (optimized using product_data tuples)
    inventory = {p_id: np.random.randint(50, 200) for p_id, _, _, _ in product_data}
    consecutive_rain_hours = 0
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.weekday()
        date_obj = dt.date()
        month = dt.month
        
        # Flags
        weekend_flag = 1 if day_of_week >= 5 else 0
        payday_flag = 1 if dt.day in [15, 28, 29, 30, 31] else 0
        h_type = None
        current_md = (month, dt.day)
        
        # Check store-specific holidays first
        s_holidays = store_specific_holidays.get(store_id, [])
        for h_name, start_md, end_md in s_holidays:
            if start_md <= current_md <= end_md:
                h_type = h_name
                break
                
        # If no store-specific holiday, check global holidays
        if not h_type:
            h_type = ph_holidays.get(current_md)

        if h_type:
            holiday_flag = 1
            holiday_type = h_type
        elif date_obj in holy_week:
            holiday_flag = 1
            holiday_type = "Holy Week"
        else:
            holiday_flag = 0
            holiday_type = "None"
        nearby_event_flag = 1 if np.random.rand() < 0.03 else 0 
        
        # Weather
        base_temp = 26 + 5 * np.sin(np.pi * (hour - 6) / 12)
        temperature = base_temp + 4 * np.sin(np.pi * month / 6) + np.random.normal(0, 1.5)
        heat_index = temperature + np.random.uniform(2, 6) if temperature > 31 else temperature
        
        rain_chance = 0.5 if month in [6, 7, 8, 9] else 0.1
        rainfall = np.random.exponential(4) if np.random.rand() < rain_chance else 0.0
        flu_season = 1 if month in [6, 7, 8] else 0
        school_season = 1 if month in [8, 9] else 0

        if date_obj in typhoon_days: rainfall += np.random.uniform(10, 30)
        
        if rainfall > 0: consecutive_rain_hours += 1
        else: consecutive_rain_hours = 0

        # 1 = Typhoon arrives tomorrow, 0 = Normal
        typhoon_incoming = 1 if (date_obj + timedelta(days=1)) in typhoon_days else 0
        
        # -----------------------------------------------------------------
        # FOOT TRAFFIC: HYPERLOCAL STORE PROFILES
        # -----------------------------------------------------------------
        traffic = 40 # Base
        
        if profile['behavior'] == 'office_hotel': # Escario
            if hour in [11, 12, 13]: traffic += 180
            if hour in [17, 18, 19]: traffic += 150
            if weekend_flag: traffic *= 0.6 # Drops on weekend
            
        elif profile['behavior'] == 'bpo_graveyard': # IT Park
            if hour >= 22 or hour <= 4: traffic += 200 # Night shift spike
            if hour in [11, 12]: traffic += 100 # Day shift lunch
            if payday_flag: traffic *= 1.5 # Huge payday bumps
            
        elif profile['behavior'] == 'student_nightlife': # Gen Maxilom
            if 10 <= hour <= 16 and not weekend_flag: traffic += 150 # Students
            if weekend_flag and (hour >= 20 or hour <= 2): traffic += 250 # Nightlife
            
        elif profile['behavior'] == 'commuter_weekend': # SRP-Talisay
            if hour in [6, 7, 8]: traffic += 200 # Morning commute
            if hour in [17, 18, 19, 20]: traffic += 220 # Evening return
            if weekend_flag and 10 <= hour <= 18: traffic += 180 # Family outings
            
        elif profile['behavior'] == 'market_mass': # Tabunok
            if hour in [4, 5, 6, 7]: traffic += 250 # Early market
            if hour in [16, 17, 18]: traffic += 200 # Late afternoon
            
        elif profile['behavior'] == 'factory_shift': # MEZ Highway
            if hour in [5, 6, 13, 14, 21, 22]: traffic += 220 # Shift changes
            
        elif profile['behavior'] == 'airport_premium': # Marina Mall
            traffic += 80 # Higher base, constant flow
            if hour in [11, 12, 13, 18, 19]: traffic += 100 # Flight peaks
        
        if rainfall > 5: traffic *= 0.6
        if nearby_event_flag: traffic *= 1.4
        foot_traffic = max(5, int(traffic + np.random.normal(0, 15)))

        temp_val = np.nan if np.random.rand() < 0.001 else round(temperature, 2)
        heat_val = np.nan if np.random.rand() < 0.001 else round(heat_index, 2)

        # Generate weather forecasts mimicking 12-24 hours in advance (with noise)
        if np.isnan(temp_val):
            forecasted_temperature = np.nan
        else:
            forecasted_temperature = round(temp_val + np.random.normal(0, 1.5), 2)
            
        if np.isnan(heat_val):
            forecasted_heat_index = np.nan
        else:
            forecasted_heat_index = round(heat_val + np.random.normal(0, 2.0), 2)
            
        if rainfall > 0:
            forecasted_rainfall = round(max(0.0, rainfall * np.random.uniform(0.6, 1.4) + np.random.normal(0, 1.0)), 2)
        else:
            forecasted_rainfall = round(np.random.exponential(1.5) if np.random.rand() < 0.15 else 0.0, 2)

        # -----------------------------------------------------------------
        # PRODUCT DEMAND: HYPERLOCAL MODIFIERS
        # -----------------------------------------------------------------
        # Optimized inner loop by iterating directly over the tuples list product_data
        for p_id, cat, p_name, price in product_data:
            promo_flag = 1 if np.random.rand() < 0.04 else 0
            
            demand = 0.05
            
            # Helper to safely check substrings
            def p_in(substrings, name):
                return any(sub in name for sub in substrings)

            # -----------------------------------------------------------------
            # 6 HYPERLOCAL DEMAND PROFILES
            # -----------------------------------------------------------------
            # Profile 1: The Fast-Moving Morning & Evening Rush
            if p_in(["3-in-1 Iced Coffee", "Energy Coffee", "Pork Siomao", "Big Bite Classic Hotdog", "Brewed Coffee", "Premium Latte", "Crunch Time Fried Chicken", "Pepperoni Pizza"], p_name):
                demand = np.random.uniform(1.5, 4.0)
                if 6 <= hour <= 9: demand *= 3.5
                elif 16 <= hour <= 20: demand *= 3.0
                elif 1 <= hour <= 4: demand *= 0.1
                
                if rainfall > 0:
                    if p_in(["Brewed Coffee", "Pork Siomao"], p_name): demand *= 2.0
                    if p_in(["Iced Coffee"], p_name): demand *= 0.3
                    
            # Profile 2: The Extreme Heat Wave Responders
            elif p_in(["Bottled Distilled Water (6L)", "Iced Coffee (16oz)", "Slurpee", "Gatorade", "Big Gulp", "Sunblock", "Biore UV", "Cooling Talcum Powder", "Rechargeable Emergency Fan"], p_name):
                demand = np.random.uniform(0.5, 1.5)
                if rainfall == 0 and temperature >= 35:
                    if 10 <= hour <= 16:
                        if p_in(["Slurpee", "Big Gulp", "Gatorade", "Sunblock", "Biore UV"], p_name): demand *= 5.0
                    if p_in(["Bottled Distilled Water"], p_name): demand *= 3.0
                
                if nearby_event_flag:
                    if p_in(["Gatorade", "Water"], p_name): demand *= 6.0
                    
            # Profile 3: The Student & Office Stationery Baseline
            elif p_in(["Intermediate Pad", "Plastic Envelope", "Black Ballpen"], p_name):
                demand = np.random.uniform(0.05, 0.2)
                if 7 <= hour <= 17: demand *= 4.0
                else: demand = 0.0
                
                if day_of_week <= 4: demand *= 2.0
                else: demand *= 0.1
                
            # Profile 4: The Weekend & BPO Graveyard Party
            elif p_in(["San Miguel", "Heineken", "Corona Extra", "Red Horse", "Soda 1.5L", "7-Select Kettle Chips", "Lay's Sour Cream", "Snickers"], p_name):
                demand = np.random.uniform(0.5, 1.5)
                is_late_night = (hour >= 21 or hour <= 3)
                is_fri_sat_night = (day_of_week == 4 and hour >= 18) or (day_of_week == 5) or (day_of_week == 6 and hour <= 3)
                
                if is_fri_sat_night: demand *= 3.0
                if is_late_night:
                    if is_fri_sat_night:
                        demand *= 4.0
                    elif profile['behavior'] == 'bpo_graveyard':
                        demand *= 2.5
                    
            # Profile 5: The "Typhoon & Blackout" Emergency Toolkit
            elif p_in(["Sachet Umbrella", "Emergency Candles", "D-Cell Batteries", "AA Energizer", "Power Bank", "Paracetamol", "Advil", "Instant Noodles", "Canned Sardines", "Canned Tuna"], p_name):
                demand = np.random.uniform(0.1, 0.3)
                if typhoon_incoming or date_obj in typhoon_days or rainfall > 0:
                    if p_in(["Umbrella", "Candles"], p_name): demand *= 15.0
                    if p_in(["Instant Noodles", "Sardines", "Batteries", "AA Energizer", "D-Cell"], p_name): demand *= 5.0
                    if p_in(["Paracetamol"], p_name): demand *= 3.0
                    
            # Profile 6: The Household Staple & Grocery Fillers
            elif p_in(["Spam", "Luncheon Meat", "All-Purpose Cream", "Condensed Milk", "Cetaphil", "Colgate", "Bathroom Tissue"], p_name):
                demand = np.random.uniform(0.2, 0.5)
                if month == 12 and 20 <= dt.day <= 24:
                    if p_in(["Cream", "Condensed"], p_name): demand *= 10.0

            #------------------------------#
            # ----- Additional cases ------#
            #------------------------------#
            # FIESTA + PAYDAY COMBINATION 
            if nearby_event_flag and payday_flag:
                if p_name in ["Soda", "Condensed Milk (300ml)", "All-Purpose Cream (250ml)", "Luncheon Meat (340g)"]:
                    demand *= 5.0  # Massive 500% spike for party preparations!.

            # PROLONGED RAIN + HUMIDITY (Health/Dengue Risk) 
            if consecutive_rain_hours > 6 and heat_index > 30:
                if p_name in ["Anti-Dengue Lotion (Sachet)", "Paracetamol (Biogesic)", "Oral Rehydration Salts"]:
                    demand *= 4.5  # 450% spike as people get sick or protect themselves from mosquitos

            # BROWNOUT PREPARATION (NGCP Warning + Heat) <---
            if date_obj in grid_warning_days and heat_index > 33:
                if p_name in ["Power Bank (10k mAh)", "Anker PowerCore 10K Power Bank", "Rechargeable Emergency Fan", "Long-life UHT Milk", "Pre-paid Load Cards"]:
                    demand *= 4.8  # 480% spike to survive a hot, powerless afternoon

            # RUSH HOUR FAST MOVERS (Commuter Essentials) <---
            # (Removed: Pork Siomao and Big Bite Classic Hotdog are already covered by Profile 1. Pocket Tissues and Menthol Candies are not in the catalog)

            # RAINY FLU SEASON (Protection & Prevention) <---
            if flu_season and rainfall > 0:
                if p_name in ["Vitamin C Syrup (Pediatric)"]:
                    demand *= 3.8  # 380% spike as parents protect their kids from getting sick in the rain

            # HOLY WEEK / LENTEN SEASON (Meatless Cultural Shift) <---
            if holiday_type == "Holy Week":
                # Spikes for meat alternatives and easy carbs
                if p_name in ["Canned Tuna (Flakes in Oil)", "Instant Pancit Canton (Calamansi)", "Century Eggs (2-Pack)", "SkyFlakes Crackers (Pack)"]:
                    demand *= 3.5  
                # CRITICAL: Cannibalization of meat products!
                elif cat == "Hot Food" and ("Pork" in p_name or "Hotdog" in p_name) or p_name in ["Spam Less Sodium 12oz", "Luncheon Meat (340g)"]:
                    demand *= 0.2  # Demand crashes by 80%
            # SCHOOL SEASON (Back-to-School Rush) <---
            if school_season:
                if p_name in ["Intermediate Pad (Sachet)", "Black Ballpen (Single)", "Plastic Envelope (Long)"]:
                    demand *= 4.0  # 400% spike as students/parents scramble for supplies
                elif cat == "Stationery":
                    demand *= 2.0  # General 200% bump for any other school items            

            if promo_flag: demand *= 1.5
            if payday_flag: demand *= 1.2
            
            demand *= np.random.uniform(0.8, 1.2)
            
            # Inventory Logic
            # Deliveries arrive daily at 4 AM. We replenish up to a capacity of 100-150 if stock is low (below 30).
            # There is a 5% chance of delivery failure/delay.
            if hour == 4 and inventory[p_id] < 30 and np.random.rand() >= 0.05:
                inventory[p_id] = np.random.randint(100, 150)
                
            intended_qty = int(round(demand))
            stockout_flag = 0
            if intended_qty > inventory[p_id]:
                qty_sold = inventory[p_id]
                stockout_flag = 1
            else:
                qty_sold = intended_qty
                
            inventory[p_id] -= qty_sold

            data_rows.append((
                dt.strftime('%Y-%m-%d %H:%M:%S'), store_id, profile['type'], profile['name'],
                p_id, p_name, cat, price, inventory[p_id], qty_sold,
                hour, day_of_week, weekend_flag, holiday_flag, holiday_type, 
                payday_flag, promo_flag, forecasted_temperature, forecasted_rainfall, forecasted_heat_index,
                foot_traffic, nearby_event_flag, stockout_flag
            ))
            
    # Process store data to add lags and true_demand before appending
    store_df = pd.DataFrame(data_rows, columns=store_columns)
    
    # 1. Un-censor demand (true_demand)
    # Identify non-stockout periods for calculating historical averages
    no_stockout = store_df[(store_df['stockout_flag'] == 0) & (store_df['inventory'] > 0)]
    hist_avg = no_stockout.groupby(['product_id', 'hour_of_day', 'day_of_week'])['qty_sold'].mean().reset_index()
    hist_avg.rename(columns={'qty_sold': 'hist_avg_sales'}, inplace=True)
    
    store_df = store_df.merge(hist_avg, on=['product_id', 'hour_of_day', 'day_of_week'], how='left')
    fallback_avg = store_df.groupby('product_id')['qty_sold'].mean().reset_index()
    fallback_avg.rename(columns={'qty_sold': 'fallback_sales'}, inplace=True)
    store_df = store_df.merge(fallback_avg, on='product_id', how='left')
    
    hist_mean = store_df['hist_avg_sales'].fillna(store_df['fallback_sales']).fillna(0)
    cond_true_demand = (store_df['stockout_flag'] == 0) & (store_df['inventory'] > 0)
    store_df['true_demand'] = np.where(cond_true_demand, store_df['qty_sold'], hist_mean.round().astype(int))
    store_df.drop(columns=['hist_avg_sales', 'fallback_sales'], inplace=True)
    
    # 2. Sort by product_id and timestamp to ensure correct lags
    store_df.sort_values(by=['product_id', 'timestamp'], inplace=True)
    
    # 3. Calculate lags per product
    product_gp = store_df.groupby('product_id')
    store_df['qty_sold_lag_1h'] = product_gp['qty_sold'].shift(1)
    store_df['qty_sold_lag_24h'] = product_gp['qty_sold'].shift(24)
    store_df['qty_sold_lag_168h'] = product_gp['qty_sold'].shift(168)
    
    store_df['foot_traffic_lag_1h'] = product_gp['foot_traffic'].shift(1)
    store_df['foot_traffic_lag_2h'] = product_gp['foot_traffic'].shift(2)
    
    # Reorder columns to match the output specification
    store_df = store_df[final_columns]
    
    # Append store's data to file
    store_df.to_csv(filename, mode='a', header=False, index=False)
    print(f"   Saved {len(store_df)} processed rows for {profile['name']}.")

# =========================================================
# 5. GLOBAL HISTORICAL ENCODERS (Second Pass)
# =========================================================
print("Calculating global store and product historical average sales...")
# Read only the necessary columns to save memory
df_stats = pd.read_csv(filename, usecols=['store_id', 'product_id', 'qty_sold'])
store_avg = df_stats.groupby('store_id')['qty_sold'].mean().to_dict()
product_avg = df_stats.groupby('product_id')['qty_sold'].mean().to_dict()
del df_stats # free memory

print("Writing final encoded columns to dataset...")
temp_filename = filename + '.tmp'
chunksize = 100000
first = True
for chunk in pd.read_csv(filename, chunksize=chunksize):
    chunk['store_historical_avg_sales'] = chunk['store_id'].map(store_avg).round(4)
    chunk['product_historical_avg_sales'] = chunk['product_id'].map(product_avg).round(4)
    chunk.to_csv(temp_filename, mode='a', index=False, header=first)
    first = False

os.replace(temp_filename, filename)
print(f"\nSUCCESS! Full multi-store dataset with advanced features saved to {filename}")