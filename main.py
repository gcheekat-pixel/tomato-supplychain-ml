import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# SECTION 1: PREPROCESSING & DATA INTEGRATION
# =====================================================================
base_dir = os.getcwd()  # Dynamic path for GitHub Actions environment
price_dir = os.path.join(base_dir, "2_Price_Market")
weather_dir = os.path.join(base_dir, "1_Weather")
holiday_dir = os.path.join(base_dir, "3_Holidays")

plots_dir = os.path.join(base_dir, "Project_Plots")
os.makedirs(plots_dir, exist_ok=True)

print("Parsing raw APMC files by individual market name...")

valid_markets = [
    'Kalikiri APMC', 'Madanapalli APMC', 'Palamaner APMC',
    'Bowenpally APMC', 'Chevella APMC', 'Gudimalkapur APMC', 
    'Mahboob Manison APMC', 'Mahboob Mansion APMC', 'Shadnagar APMC'
]

def parse_agmarknet_folder(folder_path, state_name):
    all_files = glob.glob(os.path.join(folder_path, "*.xlsx")) + glob.glob(os.path.join(folder_path, "*.xls"))
    records = []
    
    for f in all_files:
        try:
            raw_df = pd.read_excel(f, header=None)
        except Exception:
            continue
            
        current_market = None
        
        for idx, row in raw_df.iterrows():
            first_cell = str(row.iloc[0]).strip()
            
            if "Market Name :" in first_cell:
                current_market = first_cell.replace("Market Name :", "").strip()
                continue
                
            if current_market and any(m.lower() in current_market.lower() for m in valid_markets):
                date_val = row.iloc[0]
                try:
                    date_dt = pd.to_datetime(date_val, dayfirst=True)
                    arrival = float(str(row.iloc[1]).replace(',', ''))
                    min_p = float(str(row.iloc[3]).replace(',', ''))
                    max_p = float(str(row.iloc[4]).replace(',', ''))
                    modal_p = float(str(row.iloc[5]).replace(',', ''))
                    
                    records.append({
                        'State': state_name,
                        'Region_Market': current_market,
                        'Date': date_dt,
                        'Arrivals_Tonnes': arrival,
                        'Min_Price': min_p,
                        'Max_Price': max_p,
                        'Modal_Price': modal_p
                    })
                except Exception:
                    continue
                    
    return pd.DataFrame(records)

ap_folder = os.path.join(price_dir, "tomato prices in ap")
tg_folder = os.path.join(price_dir, "tomato prices in telangana")

df_ap = parse_agmarknet_folder(ap_folder, "Andhra Pradesh")
df_tg = parse_agmarknet_folder(tg_folder, "Telangana")

df_market = pd.concat([df_ap, df_tg], ignore_index=True)
df_market['Region_Market'] = df_market['Region_Market'].replace({'Mahboob Manison APMC': 'Mahboob Mansion APMC'})

master_daily_path = os.path.join(base_dir, "Master_Tomato_Supply_Chain_Data.xlsx")
master_df = pd.read_excel(master_daily_path)
master_df['Date'] = pd.to_datetime(master_df['Date'])

weather_holiday_cols = [
    'State', 'Date', 'Avg_Temp_C', 'Max_Temp_C', 'Min_Temp_C', 
    'Daily_Rainfall_mm', 'Is_Holiday'
]
ref_weather = master_df[weather_holiday_cols].drop_duplicates(subset=['State', 'Date'])

df_merged = pd.merge(df_market, ref_weather, on=['State', 'Date'], how='left')

agg_rules = {
    'Arrivals_Tonnes': 'sum',
    'Modal_Price': 'mean',
    'Min_Price': 'min',
    'Max_Price': 'max',
    'Avg_Temp_C': 'mean',
    'Max_Temp_C': 'max',
    'Min_Temp_C': 'min',
    'Daily_Rainfall_mm': 'sum',
    'Is_Holiday': 'sum'
}

weekly_dfs = []
for (state, market), group in df_merged.groupby(['State', 'Region_Market']):
    g = group.sort_values('Date').set_index('Date')
    w_df = g.resample('W-SUN').agg(agg_rules).reset_index()
    w_df['State'] = state
    w_df['Region_Market'] = market
    w_df['Year'] = w_df['Date'].dt.isocalendar().year
    w_df['Week_Number'] = w_df['Date'].dt.isocalendar().week
    
    rename_cols = {
        'Date': 'Week_Ending_Date',
        'Arrivals_Tonnes': 'Weekly_Arrivals_Tonnes',
        'Modal_Price': 'Avg_Modal_Price',
        'Min_Price': 'Min_Price_Week',
        'Max_Price': 'Max_Price_Week',
        'Avg_Temp_C': 'Weekly_Avg_Temp_C',
        'Max_Temp_C': 'Weekly_Max_Temp_C',
        'Min_Temp_C': 'Weekly_Min_Temp_C',
        'Daily_Rainfall_mm': 'Weekly_Total_Rain_mm',
        'Is_Holiday': 'Holidays_In_Week'
    }
    w_df = w_df.rename(columns=rename_cols)
    
    w_df['Price_Lag_1W'] = w_df['Avg_Modal_Price'].shift(1)
    w_df['Price_Lag_2W'] = w_df['Avg_Modal_Price'].shift(2)
    w_df['Arrivals_Lag_1W'] = w_df['Weekly_Arrivals_Tonnes'].shift(1)
    w_df['Rain_Lag_1W'] = w_df['Weekly_Total_Rain_mm'].shift(1)
    
    weekly_dfs.append(w_df)

final_weekly_df = pd.concat(weekly_dfs, ignore_index=True)

output_path = os.path.join(base_dir, "Weekly_Tomato_Supply_Chain_Data.xlsx")
final_weekly_df.to_excel(output_path, index=False)
print("Preprocessing complete. Weekly matrix generated.")

# =====================================================================
# SECTION 2: MACHINE LEARNING MODEL EVALUATION
# =====================================================================
df = pd.read_excel(output_path)
df_clean = df.dropna().copy().sort_values(by=['Region_Market', 'Week_Ending_Date']).reset_index(drop=True)

feature_cols = [
    'Week_Number', 'Weekly_Avg_Temp_C', 'Weekly_Max_Temp_C',
    'Weekly_Total_Rain_mm', 'Holidays_In_Week',
    'Price_Lag_1W', 'Price_Lag_2W', 'Arrivals_Lag_1W', 'Rain_Lag_1W'
]
target_col = 'Avg_Modal_Price'

hub_df = df_clean[df_clean['Region_Market'] == 'Madanapalli APMC'].reset_index(drop=True)

X = hub_df[feature_cols]
y = hub_df[target_col]

split_idx = int(len(hub_df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
    "KNN Regressor": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))
    ])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE (Rs/Qtl)': round(mae, 2),
        'RMSE (Rs/Qtl)': round(rmse, 2),
        'MAPE (%)': round(mape, 2),
        'R2 Score': round(r2, 4)
    })

scorecard = pd.DataFrame(results)
metrics_file = os.path.join(base_dir, "Model_Evaluation_Metrics.xlsx")
scorecard.to_excel(metrics_file, index=False)
print("ML training complete. Metrics recorded.")

# =====================================================================
# SECTION 3: ANYLOGISTIX MULTI-SHEET SCENARIO EXPORT
# =====================================================================
demand_records = []
for market_name, m_group in df_clean.groupby('Region_Market'):
    m_sorted = m_group.sort_values('Week_Ending_Date').reset_index(drop=True)
    X_m = m_sorted[feature_cols]
    y_m = m_sorted['Weekly_Arrivals_Tonnes']
    
    rf_m = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_m.fit(X_m, y_m)
    m_sorted['Demand_Tonnes'] = rf_m.predict(X_m).round(2)
    
    for _, row in m_sorted.iterrows():
        demand_records.append({
            'Customer': row['Region_Market'],
            'Product': 'Tomato',
            'Demand Type': 'Periodic Demand',
            'Parameters': row['Demand_Tonnes'],
            'Time Period': row['Week_Ending_Date'].strftime('%Y-%m-%d'),
            'Revenue_Per_Tonne_Rs': round(row['Avg_Modal_Price'] * 10, 2)
        })

df_demand = pd.DataFrame(demand_records)

# Mandatory scenario topology tables for AnyLogistix
df_customers = pd.DataFrame([
    {"Name": "Madanapalli APMC", "Type": "Customer", "Location": "13.5500, 78.5000"},
    {"Name": "Palamaner APMC", "Type": "Customer", "Location": "13.2000, 78.7500"},
    {"Name": "Kalikiri APMC", "Type": "Customer", "Location": "13.6800, 78.7800"},
    {"Name": "Bowenpally APMC", "Type": "Customer", "Location": "17.4700, 78.4800"},
    {"Name": "Chevella APMC", "Type": "Customer", "Location": "17.3100, 78.1300"},
    {"Name": "Gudimalkapur APMC", "Type": "Customer", "Location": "17.3800, 78.4400"},
    {"Name": "Mahboob Mansion APMC", "Type": "Customer", "Location": "17.3600, 78.5000"},
    {"Name": "Shadnagar APMC", "Type": "Customer", "Location": "17.0700, 78.2000"}
])

df_products = pd.DataFrame([
    {"Name": "Tomato", "Unit": "Tonnes", "Selling Price": 25000}
])

df_expenses = pd.DataFrame([
    {"Facility": "Regional Supply Hub", "Expense Type": "Initial Cost", "Value": 10000}
])

df_paths = pd.DataFrame([
    {"From": "Madanapalli APMC", "To": "Bowenpally APMC", "Product": "Tomato", "Transportation Policy": "LTL"}
])

# Write all 5 required tables into a single Excel scenario workbook
alx_path = os.path.join(base_dir, "anyLogistix_Scenario_Input.xlsx")
with pd.ExcelWriter(alx_path, engine="openpyxl") as writer:
    df_customers.to_excel(writer, sheet_name="Customers", index=False)
    df_products.to_excel(writer, sheet_name="Products", index=False)
    df_demand.to_excel(writer, sheet_name="Demand", index=False)
    df_expenses.to_excel(writer, sheet_name="Facility Expenses", index=False)
    df_paths.to_excel(writer, sheet_name="Paths", index=False)

print("Master pipeline execution complete. 'anyLogistix_Scenario_Input.xlsx' generated successfully.")