import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import probplot
from statsmodels.tsa.stattools import acf

file_name = r"D:\office\Projects\python\Forcasting\goldprice_final.xlsx"
order = (6, 2, 0)


TRAIN_SHEET = 'Gold Price Data (2005-2024)'

FULL_SHEET = 'Gold Price Data (2005-2025)'

print(f"Start ARIMA{order}...")
print("-" * 40)


try:
    df_train = pd.read_excel(file_name, sheet_name=TRAIN_SHEET)
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    df_train.set_index('Date', inplace=True)
    train_data = df_train['Unadjust Value($)'].dropna()
    
    df_full = pd.read_excel(file_name, sheet_name=FULL_SHEET)
    df_full['Date'] = pd.to_datetime(df_full['Date'])
    df_full.set_index('Date', inplace=True)
    full_data = df_full['Unadjust Value($)'].dropna()

    test_start_date = train_data.index[-1] + pd.DateOffset(months=1)
    

    test_data = full_data[full_data.index >= test_start_date]
    test_size = len(test_data)
    forecast_steps = test_size 

    print(f"Uploaded. Train from '{TRAIN_SHEET}' ({len(train_data)} points).")
    print(f"Test from real data '{FULL_SHEET}' ({test_size} points).")
    print(f"Test cycle: from {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")

except FileNotFoundError:
    print(f"Path not found: {file_name}")
    exit()
except Exception as e:
    print(f"Error while reading {e}")
    exit()


try:

    print("\n--- Train (2005-2024) ---")
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    fitted_values = model_fit.fittedvalues

    forecast_test = model_fit.get_prediction(start=test_data.index[0], end=test_data.index[-1])
    predicted_test_values = forecast_test.predicted_mean
    

    rmse = np.sqrt(mean_squared_error(test_data, predicted_test_values))
    
    print("\n--- Accuracy evaluate ---")
    print(f"**RMSE from real data ({test_size} months): {rmse:.2f}**")
    
    confidence_intervals = forecast_test.conf_int()

    print(f"Predict successfully next {test_size} months.")

except Exception as e:
    print(f"\nFailed to build ARIMA model: {e}")
    exit()

try:
    
    residuals = model_fit.resid
    std_dev = np.sqrt(model_fit.scale)
    standardized_residuals = residuals / std_dev
    
    
    residual_df = pd.DataFrame({
        'Residuals': residuals,
        'Standardized_Residuals': standardized_residuals 
    })

    
    (osr, score), (slope, intercept, r) = probplot(standardized_residuals, plot=None)
    qq_df = pd.DataFrame({
        'Theoretical_Quantiles': score,
        'Sample_Quantiles': osr
    })

    
    hist_counts, hist_bins = np.histogram(standardized_residuals, bins='auto', density=True)
    hist_df = pd.DataFrame({
        'Bin_Start': hist_bins[:-1],
        'Bin_End': hist_bins[1:],
        'Density': hist_counts
    })

    
    LAG_COUNT = 20
    
    acf_values, conf_int = acf(standardized_residuals, nlags=LAG_COUNT, alpha=0.05, fft=False)
    lags = np.arange(len(acf_values))
    
    acf_df = pd.DataFrame({
        'Lag': lags,
        'ACF_Value': acf_values,
        'Lower_CI': conf_int[:, 0], 
        'Upper_CI': conf_int[:, 1]
    })
    
    
    forecast_df = pd.DataFrame({
        'Predicted_Value': predicted_test_values,
        'Actual_Value': test_data,
        'Lower_CI_95': confidence_intervals.iloc[:, 0],
        'Upper_CI_95': confidence_intervals.iloc[:, 1]
    })
    
    
    export_file = "arima_diagnostic_export.xlsx"
    with pd.ExcelWriter(export_file) as writer:
        residual_df.to_excel(writer, sheet_name='Residuals_TimeSeries', index=True, index_label='Date')
        qq_df.to_excel(writer, sheet_name='QQ_Plot_Data', index=False)
        hist_df.to_excel(writer, sheet_name='Histogram_Data', index=False)
        acf_df.to_excel(writer, sheet_name='ACF_Data', index=False)
        forecast_df.to_excel(writer, sheet_name='Out_of_Sample_Forecast', index=True, index_label='Date')

    print(f"\nSuccessfully exported ALL data to a single Excel file: {export_file}")

except Exception as e:
    print(f"\nFailed to export diagnostic or forecast data: {e}")


print("\n--- Diagnostic Plots ---")

LAG_COUNT = 20 
print(f"Plotting diagnostics with Lags for ACF: {LAG_COUNT}")

fig = plt.figure(figsize=(15, 12))
fig.suptitle(f'Diagnostic Plots ARIMA{order} (Lags for ACF: {LAG_COUNT})', fontsize=16)

model_fit.plot_diagnostics(fig=fig, lags=LAG_COUNT)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


try:
    ljung_box_result = model_fit.test_serial_correlation(method='ljungbox', lags=10)
    p_value = ljung_box_result[0][0][2] 
    print(f"\nLjung-Box evaluate (Lag 10):")
    print(f"  P-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  Good Result: Residue is white noise (Good model).")
    else:
        print("  Bad Result: Adjust p, d, q.")
except Exception as e:
    print(f"Cannot run Ljung-Box test: {e}")


print("\n--- Plotting (Out-of-Sample) ---")
plt.figure(figsize=(14, 7))


plt.plot(train_data, label='Train data (2005-2024)', color='blue')


plt.plot(test_data, label='Real data (2025)', color='green', linewidth=2)

plt.plot(predicted_test_values, label=f'Out-of-Sample ARIMA{order}', color='red', linestyle='--', linewidth=2)

plt.plot(fitted_values, label=f'Fit value{order}', color='black', linestyle=':', linewidth=1.5)

plt.fill_between(
    confidence_intervals.index,
    confidence_intervals.iloc[:, 0],
    confidence_intervals.iloc[:, 1],
    color='red', alpha=0.15, label='Confident interval 95%'
)

plt.title(f'Gold price (Out-of-Sample): RMSE={rmse:.2f} | ARIMA{order}', fontsize=16)
plt.xlabel('Time')
plt.ylabel('Value (USD)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.show()

print(model_fit.summary())

print("-" * 40)
print("Completed Out-of-Sample analytic.")