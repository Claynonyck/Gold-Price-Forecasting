import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
from statsmodels.tsa.stattools import acf

EXCEL_FILE = r"D:\office\Projects\python\Forcasting\goldprice_final.xlsx"
TRAIN_SHEET = 'Gold Price Data (2005-2024)'
FULL_SHEET = 'Gold Price Data (2005-2025)'

TARGET_COL = 'Unadjust Value($)'
EXOG_COLS = ['Interest rate', 'Consumer price index', 'Sticky consumer price index']

USE_LOG = True
SCALE_VARS = True
PRIOR_SCALE = 0.5
REGRESSOR_MODE = 'multiplicative'

print("--- Step 1: Loading & Preparing Data (Quick Start Style) ---")

try:
    df_train_raw = pd.read_excel(EXCEL_FILE, sheet_name=TRAIN_SHEET)
    df_full_raw = pd.read_excel(EXCEL_FILE, sheet_name=FULL_SHEET)
except FileNotFoundError:
    print(f"Error: The file '{EXCEL_FILE}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    sys.exit(1)

df_train_raw.columns = df_train_raw.columns.str.strip()
df_full_raw.columns = df_full_raw.columns.str.strip()

df = df_train_raw.rename(columns={'Date': 'ds', TARGET_COL: 'y'})
df['ds'] = pd.to_datetime(df['ds'])

test_start_date = df['ds'].iloc[-1]
mask_test = pd.to_datetime(df_full_raw['Date']) > test_start_date
df_test = df_full_raw[mask_test].copy()

future = df_test.rename(columns={'Date': 'ds', TARGET_COL: 'y'})
future['ds'] = pd.to_datetime(future['ds'])

if future.empty:
    print("Warning: No future data found for testing. Check date ranges.")
    sys.exit(1)

if USE_LOG:
    print("Applying Log Transformation...")
    df['y'] = np.log(df['y'])

if SCALE_VARS:
    print("Scaling Regressors...")
    full_exog = pd.concat([df[EXOG_COLS], future[EXOG_COLS]])
    scaler = StandardScaler()
    scaler.fit(full_exog)
    
    df[EXOG_COLS] = scaler.transform(df[EXOG_COLS])
    future[EXOG_COLS] = scaler.transform(future[EXOG_COLS])

print("--- Step 2: Fitting Model ---")
m = Prophet(
    yearly_seasonality=10, 
    seasonality_mode='multiplicative',
    changepoint_prior_scale=PRIOR_SCALE,
    seasonality_prior_scale=10.0
)

for col in EXOG_COLS:
    m.add_regressor(col, mode=REGRESSOR_MODE)

m.fit(df)
forecast_train = m.predict(df)

print("--- Step 3: Predicting ---")
forecast = m.predict(future)

y_pred = forecast['yhat'].values
y_true = df_test[TARGET_COL].values 

if USE_LOG:
    y_pred = np.exp(y_pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"\nResults:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

print("\n--- Exporting Diagnostic Data ---")
    
residuals = y_true - y_pred
residual_df = pd.DataFrame({
    'Date': future['ds'],
    'Residuals': residuals
})

LAG_COUNT = 15 
acf_values, conf_int = acf(residuals, nlags=LAG_COUNT, alpha=0.05, fft=False)
lags = np.arange(len(acf_values))

acf_df = pd.DataFrame({
    'Lag': lags,
    'ACF_Value': acf_values,
    'Lower_CI': conf_int[:, 0], 
    'Upper_CI': conf_int[:, 1]
})

forecast_df = pd.DataFrame({
    'Date': future['ds'],
    'Predicted_Value': y_pred,
    'Actual_Value': y_true,
    'Lower_CI_95': np.exp(forecast['yhat_lower'].values) if USE_LOG else forecast['yhat_lower'].values,
    'Upper_CI_95': np.exp(forecast['yhat_upper'].values) if USE_LOG else forecast['yhat_upper'].values
})

export_file = "prophet_diagnostic_export.xlsx"
with pd.ExcelWriter(export_file) as writer:
    residual_df.to_excel(writer, sheet_name='Residuals_TimeSeries', index=False)
    acf_df.to_excel(writer, sheet_name='ACF_Data', index=False)
    forecast_df.to_excel(writer, sheet_name='Out_of_Sample_Forecast', index=False)

print(f"\nSuccessfully exported ALL data to a single Excel file: {export_file}")

print("--- Step 4: Plotting ---")

fig1 = m.plot_components(forecast, figsize=(10, 8))
fig1.suptitle('Prophet Model Components', fontsize=16, y=1.02)
plt.show() 

plt.figure(figsize=(12, 7))

df_plot = df.copy()
fitted_values = forecast_train['yhat'].values
if USE_LOG:

    df_plot['y'] = np.exp(df_plot['y'])
    fitted_values = np.exp(fitted_values)

plt.plot(pd.to_datetime(df_plot['ds']), df_plot['y'], 
         label='History (Train)', color='#6c757d', alpha=0.8, linewidth=1.5)
plt.plot(pd.to_datetime(df_plot['ds']), fitted_values,
         label=f'In-Sample Fit', color='orange', linestyle=':', linewidth=1.5)
plt.axvline(x=pd.to_datetime(future['ds'].iloc[0]), color='black', linestyle='-', linewidth=1.0, alpha=0.6)

plt.plot(pd.to_datetime(future['ds']), y_true, 
         label='Actual (Test)', color='red', linewidth=3) 

plt.plot(pd.to_datetime(future['ds']), y_pred, 
         label='Forecast', color='blue', linestyle='--', linewidth=3) 


forecast_interval = forecast.copy()
if USE_LOG:
    forecast_interval['yhat_lower'] = np.exp(forecast_interval['yhat_lower'])
    forecast_interval['yhat_upper'] = np.exp(forecast_interval['yhat_upper'])

plt.fill_between(pd.to_datetime(future['ds']), 
                 forecast_interval['yhat_lower'], 
                 forecast_interval['yhat_upper'], 
                 color='blue', alpha=0.15, label='Uncertainty Interval')

plt.title(f'Gold Price Forecast vs Actuals (RMSE: {rmse:.2f} | MAE: {mae:.2f})', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel(f'{TARGET_COL}', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout() 
plt.show()


residuals = y_true - y_pred

plt.figure(figsize=(10, 4))
plt.bar(pd.to_datetime(future['ds']), residuals, color='purple', width=20, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals (Error) Over Test Period')
plt.xlabel('Date')
plt.ylabel('Error (Actual - Predicted) $')
plt.grid(True, axis='y', alpha=0.3)
plt.show()

print("Completed Prophet quick-start analysis with three diagnostic plots.")