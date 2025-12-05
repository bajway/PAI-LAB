import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 50

screen_on_time = np.random.uniform(1, 10, n_samples)
battery_life = 48 - 4.2 * screen_on_time + np.random.normal(0, 2, n_samples)

df = pd.DataFrame({
    'ScreenOnTime': screen_on_time,
    'BatteryLife': battery_life
})

print("Dataset Preview:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")

X = df[['ScreenOnTime']]
y = df['BatteryLife']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
slope = model.coef_[0]

print(f"\n--- Linear Regression Results ---")
print(f"Intercept: {intercept:.4f}")
print(f"Slope: {slope:.4f}")

r2 = model.score(X, y)
print(f"R² Value: {r2:.4f}")

correlation, p_value_corr = stats.pearsonr(df['ScreenOnTime'], df['BatteryLife'])
print(f"\n--- Correlation Analysis ---")
print(f"Pearson Correlation Coefficient: {correlation:.4f}")
print(f"P-value: {p_value_corr:.6f}")

alpha = 0.05
print(f"\n--- Significance Test (α = {alpha}) ---")
if p_value_corr < alpha:
    print(f"Result: Significant correlation (p-value {p_value_corr:.6f} < {alpha})")
else:
    print(f"Result: No significant correlation (p-value {p_value_corr:.6f} >= {alpha})")

if correlation < 0 and p_value_corr < alpha:
    print("Conclusion: There IS a significant NEGATIVE correlation")
else:
    print("Conclusion: There is NO significant negative correlation")

n = len(df)
t_statistic = correlation * np.sqrt((n - 2) / (1 - correlation**2))
p_value_t = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-2))

print(f"\n--- T-Test for Correlation ---")
print(f"T-statistic: {t_statistic:.4f}")
print(f"Degrees of Freedom: {n-2}")
print(f"P-value (two-tailed): {p_value_t:.6f}")

prediction_5hrs = model.predict([[5]])[0]
prediction_8hrs = model.predict([[8]])[0]
print(f"\n--- Predictions ---")
print(f"Predicted Battery Life for 5 hrs screen time: {prediction_5hrs:.2f} hours")
print(f"Predicted Battery Life for 8 hrs screen time: {prediction_8hrs:.2f} hours")

plt.figure(figsize=(10, 6))
plt.scatter(df['ScreenOnTime'], df['BatteryLife'], alpha=0.6, label='Data Points')
plt.plot(df['ScreenOnTime'], model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel('Average Screen-On Time per Day (hours)')
plt.ylabel('Battery Life (hours before recharge)')
plt.title('Smartphone Battery Life vs Screen-On Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('battery_life_regression.png', dpi=100, bbox_inches='tight')
plt.show()

print(f"\n--- Summary Statistics ---")
print(df.describe())
