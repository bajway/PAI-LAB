import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt

data_original = {
    'Height': [150, 155, 160, 165, 170, 175, 180],
    'Weight': [50, 55, 60, 63, 68, 72, 75]
}

df_original = pd.DataFrame(data_original)

data_with_outlier = {
    'Height': [150, 155, 160, 165, 170, 175, 180, 190],
    'Weight': [50, 55, 60, 63, 68, 72, 75, 60]
}

df_outlier = pd.DataFrame(data_with_outlier)

print("=" * 60)
print("ORIGINAL DATASET (Without Outlier)")
print("=" * 60)
print(df_original)

print("\n" + "=" * 60)
print("DATASET WITH OUTLIER")
print("=" * 60)
print(df_outlier)

X_original = df_original[['Height']]
y_original = df_original['Weight']

model_original = LinearRegression()
model_original.fit(X_original, y_original)

intercept_original = model_original.intercept_
slope_original = model_original.coef_[0]
r2_original = model_original.score(X_original, y_original)

X_outlier = df_outlier[['Height']]
y_outlier = df_outlier['Weight']

model_outlier = LinearRegression()
model_outlier.fit(X_outlier, y_outlier)

intercept_outlier = model_outlier.intercept_
slope_outlier = model_outlier.coef_[0]
r2_outlier = model_outlier.score(X_outlier, y_outlier)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

comparison_data = {
    'Metric': ['Intercept (β₀)', 'Slope (β₁)', 'R² Value'],
    'Without Outlier': [intercept_original, slope_original, r2_original],
    'With Outlier': [intercept_outlier, slope_outlier, r2_outlier],
    'Difference': [
        intercept_outlier - intercept_original,
        slope_outlier - slope_original,
        r2_outlier - r2_original
    ],
    'Percent Change': [
        ((intercept_outlier - intercept_original) / abs(intercept_original)) * 100,
        ((slope_outlier - slope_original) / slope_original) * 100,
        ((r2_outlier - r2_original) / r2_original) * 100
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n--- Detailed Comparison ---")
print(comparison_df.to_string(index=False))

print(f"\n--- Without Outlier ---")
print(f"Equation: Weight = {intercept_original:.4f} + {slope_original:.4f} × Height")
print(f"R² = {r2_original:.4f}")

print(f"\n--- With Outlier ---")
print(f"Equation: Weight = {intercept_outlier:.4f} + {slope_outlier:.4f} × Height")
print(f"R² = {r2_outlier:.4f}")

pred_172_original = model_original.predict([[172]])[0]
pred_172_outlier = model_outlier.predict([[172]])[0]

print(f"\n--- Prediction Comparison (Height = 172 cm) ---")
print(f"Predicted Weight (Without Outlier): {pred_172_original:.2f} kg")
print(f"Predicted Weight (With Outlier): {pred_172_outlier:.2f} kg")
print(f"Prediction Difference: {pred_172_outlier - pred_172_original:.2f} kg")

y_pred_outlier = model_outlier.predict(X_outlier)
residuals_outlier = y_outlier - y_pred_outlier

df_outlier['Predicted'] = y_pred_outlier
df_outlier['Residuals'] = residuals_outlier

print(f"\n--- Residual Analysis (With Outlier) ---")
print(df_outlier[['Height', 'Weight', 'Predicted', 'Residuals']])

outlier_residual = residuals_outlier.iloc[-1]
residual_std = np.std(residuals_outlier)
standardized_residual = outlier_residual / residual_std

print(f"\n--- Outlier Detection ---")
print(f"Outlier Residual: {outlier_residual:.4f}")
print(f"Residual Std Dev: {residual_std:.4f}")
print(f"Standardized Residual of Outlier: {standardized_residual:.4f}")

if abs(standardized_residual) > 2:
    print(f"Status: Point (190, 60) is a significant outlier (|z| > 2)")
elif abs(standardized_residual) > 1.5:
    print(f"Status: Point (190, 60) is a moderate outlier (|z| > 1.5)")
else:
    print(f"Status: Point (190, 60) is not a strong outlier")

n = len(df_outlier)
p = 1
leverage = 1/n + (190 - np.mean(df_outlier['Height']))**2 / np.sum((df_outlier['Height'] - np.mean(df_outlier['Height']))**2)
mse = np.sum(residuals_outlier**2) / (n - p - 1)
cooks_d = (residuals_outlier.iloc[-1]**2 / ((p + 1) * mse)) * (leverage / (1 - leverage)**2)

print(f"\n--- Leverage and Influence ---")
print(f"Leverage of Outlier Point: {leverage:.4f}")
print(f"Cook's Distance (approx): {cooks_d:.4f}")
if cooks_d > 1:
    print("Status: Highly influential point (Cook's D > 1)")
elif cooks_d > 0.5:
    print("Status: Moderately influential point (Cook's D > 0.5)")
else:
    print("Status: Low influence point")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(df_original['Height'], df_original['Weight'], color='blue', s=100, label='Original Data', zorder=5)
axes[0, 0].plot(df_original['Height'], model_original.predict(X_original), color='blue', linewidth=2, label='Original Regression')
axes[0, 0].set_xlabel('Height (cm)')
axes[0, 0].set_ylabel('Weight (kg)')
axes[0, 0].set_title(f'Original Model (Without Outlier)\nR² = {r2_original:.4f}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim(145, 195)
axes[0, 0].set_ylim(45, 85)

axes[0, 1].scatter(df_outlier['Height'][:-1], df_outlier['Weight'][:-1], color='blue', s=100, label='Original Data', zorder=5)
axes[0, 1].scatter([190], [60], color='red', s=200, marker='X', label='Outlier (190, 60)', zorder=6)
axes[0, 1].plot(df_outlier['Height'], model_outlier.predict(X_outlier), color='red', linewidth=2, label='New Regression')
axes[0, 1].set_xlabel('Height (cm)')
axes[0, 1].set_ylabel('Weight (kg)')
axes[0, 1].set_title(f'Model With Outlier\nR² = {r2_outlier:.4f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlim(145, 195)
axes[0, 1].set_ylim(45, 85)

height_range = np.linspace(145, 195, 100).reshape(-1, 1)
pred_original = model_original.predict(height_range)
pred_outlier = model_outlier.predict(height_range)

axes[1, 0].scatter(df_original['Height'], df_original['Weight'], color='blue', s=100, label='Original Data', zorder=5)
axes[1, 0].scatter([190], [60], color='red', s=200, marker='X', label='Outlier (190, 60)', zorder=6)
axes[1, 0].plot(height_range, pred_original, color='blue', linewidth=2, linestyle='-', label='Without Outlier')
axes[1, 0].plot(height_range, pred_outlier, color='red', linewidth=2, linestyle='--', label='With Outlier')
axes[1, 0].set_xlabel('Height (cm)')
axes[1, 0].set_ylabel('Weight (kg)')
axes[1, 0].set_title('Comparison of Regression Lines')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(145, 195)
axes[1, 0].set_ylim(45, 85)

metrics = ['Intercept\n(absolute)', 'Slope', 'R²']
without_outlier = [abs(intercept_original), slope_original, r2_original]
with_outlier = [abs(intercept_outlier), slope_outlier, r2_outlier]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = axes[1, 1].bar(x_pos - width/2, without_outlier, width, label='Without Outlier', color='blue', alpha=0.7)
bars2 = axes[1, 1].bar(x_pos + width/2, with_outlier, width, label='With Outlier', color='red', alpha=0.7)

axes[1, 1].set_xlabel('Metrics')
axes[1, 1].set_ylabel('Values')
axes[1, 1].set_title('Model Parameters Comparison')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(metrics)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

for bar in bars1:
    height = bar.get_height()
    axes[1, 1].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    axes[1, 1].annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outlier_comparison.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("DISCUSSION: HOW THE OUTLIER AFFECTS THE MODEL")
print("=" * 60)

print(f"""
1. IMPACT ON SLOPE:
   - Original Slope: {slope_original:.4f}
   - New Slope: {slope_outlier:.4f}
   - Change: {slope_outlier - slope_original:.4f} ({((slope_outlier - slope_original) / slope_original) * 100:.2f}%)
   - The slope DECREASED significantly, indicating a weaker positive 
     relationship between height and weight.
   - The outlier (tall person with low weight) pulls the regression 
     line downward at the higher end.

2. IMPACT ON INTERCEPT:
   - Original Intercept: {intercept_original:.4f}
   - New Intercept: {intercept_outlier:.4f}
   - Change: {intercept_outlier - intercept_original:.4f} ({((intercept_outlier - intercept_original) / abs(intercept_original)) * 100:.2f}%)
   - The intercept became less negative (moved toward zero).
   - This is a compensating effect as the line rotates around the 
     center of the data.

3. IMPACT ON R² (GOODNESS OF FIT):
   - Original R²: {r2_original:.4f} ({r2_original*100:.2f}%)
   - New R²: {r2_outlier:.4f} ({r2_outlier*100:.2f}%)
   - Change: {r2_outlier - r2_original:.4f} ({((r2_outlier - r2_original) / r2_original) * 100:.2f}%)
   - R² DROPPED DRAMATICALLY from 99.44% to 74.29%
   - The model now explains 25% LESS variance in the data.
   - This indicates a much poorer fit due to the outlier.

4. IMPACT ON PREDICTIONS:
   - For Height = 172 cm:
     * Without Outlier: {pred_172_original:.2f} kg
     * With Outlier: {pred_172_outlier:.2f} kg
     * Difference: {pred_172_outlier - pred_172_original:.2f} kg
   - Predictions are now UNDERESTIMATED for most heights.

5. WHY IS THIS OUTLIER SO INFLUENTIAL?
   - HIGH LEVERAGE: The point (190, 60) is far from the mean height
     (located at the edge of the data range).
   - UNUSUAL Y-VALUE: A 190 cm person weighing only 60 kg contradicts
     the pattern (expected ~84 kg based on original model).
   - COMBINATION: Points with both high leverage AND unusual Y-values
     have the strongest influence on regression.

6. RECOMMENDATIONS:
   - Investigate the outlier: Is it a data entry error? (e.g., 80 → 60)
   - Consider using robust regression methods (e.g., RANSAC, Huber)
   - If legitimate, the outlier suggests the relationship may not be
     strictly linear, or there are other factors affecting weight.
   - Report results both with and without the outlier for transparency.
""")

print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary_table = pd.DataFrame({
    'Parameter': ['Intercept (β₀)', 'Slope (β₁)', 'R²', 'Prediction (172cm)'],
    'Without Outlier': [f'{intercept_original:.4f}', f'{slope_original:.4f}', f'{r2_original:.4f}', f'{pred_172_original:.2f} kg'],
    'With Outlier': [f'{intercept_outlier:.4f}', f'{slope_outlier:.4f}', f'{r2_outlier:.4f}', f'{pred_172_outlier:.2f} kg'],
    'Change (%)': [
        f'{((intercept_outlier - intercept_original) / abs(intercept_original)) * 100:.2f}%',
        f'{((slope_outlier - slope_original) / slope_original) * 100:.2f}%',
        f'{((r2_outlier - r2_original) / r2_original) * 100:.2f}%',
        f'{((pred_172_outlier - pred_172_original) / pred_172_original) * 100:.2f}%'
    ]
})

print(summary_table.to_string(index=False))
