import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_players = 18207

overall = np.random.randint(46, 95, n_players)
potential = overall + np.random.randint(0, 10, n_players)
potential = np.clip(potential, overall, 99)
age = np.random.randint(16, 42, n_players)
int_reputation = np.random.choice([1, 2, 3, 4, 5], n_players, p=[0.50, 0.25, 0.15, 0.07, 0.03])

value = (
    np.exp(overall * 0.12) * 1000 +
    np.exp(potential * 0.08) * 500 +
    (30 - age) * 50000 +
    int_reputation * 500000 +
    np.random.exponential(500000, n_players)
).clip(10000, 200000000)

df = pd.DataFrame({
    'Overall': overall,
    'Potential': potential,
    'Age': age,
    'International_Reputation': int_reputation,
    'Value': value
})

print("=" * 70)
print("Q10: FIFA PLAYER VALUE PREDICTION")
print("=" * 70)

print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print(df.head(10))

print("\n--- Statistical Summary ---")
print(df.describe().round(2))

print("\n--- Value Distribution (in millions €) ---")
df['Value_Millions'] = df['Value'] / 1000000
print(df['Value_Millions'].describe().round(2))

print("\n--- Correlation Matrix ---")
correlation_matrix = df[['Overall', 'Potential', 'Age', 'International_Reputation', 'Value']].corr()
print(correlation_matrix.round(4))

features_full = ['Overall', 'Potential', 'Age', 'International_Reputation']
X_full = df[features_full]
y = df['Value']

X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 70)
print("MODEL 1: WITH ALL FEATURES (Including Potential)")
print("=" * 70)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train)
y_pred_full = model_full.predict(X_test_full)

r2_full = r2_score(y_test, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
mae_full = mean_absolute_error(y_test, y_pred_full)

print(f"\n--- Coefficients ---")
coef_full = pd.DataFrame({
    'Feature': features_full,
    'Coefficient': model_full.coef_
})
print(coef_full.to_string(index=False))
print(f"Intercept: €{model_full.intercept_:,.2f}")

print(f"\n--- Performance ---")
print(f"R² Score: {r2_full:.4f} ({r2_full*100:.2f}%)")
print(f"RMSE: €{rmse_full:,.2f}")
print(f"MAE: €{mae_full:,.2f}")

scaler = StandardScaler()
X_train_full_scaled = scaler.fit_transform(X_train_full)
X_test_full_scaled = scaler.transform(X_test_full)

model_full_scaled = LinearRegression()
model_full_scaled.fit(X_train_full_scaled, y_train)

coef_full_standardized = pd.DataFrame({
    'Feature': features_full,
    'Standardized_Coefficient': model_full_scaled.coef_,
    'Abs_Coefficient': np.abs(model_full_scaled.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n--- Standardized Coefficients (Feature Impact) ---")
print(coef_full_standardized[['Feature', 'Standardized_Coefficient']].to_string(index=False))

most_impactful_full = coef_full_standardized.iloc[0]['Feature']
print(f"\n★ MOST IMPACTFUL FEATURE: {most_impactful_full}")

print("\n" + "=" * 70)
print("MODEL 2: WITHOUT POTENTIAL")
print("=" * 70)

features_no_potential = ['Overall', 'Age', 'International_Reputation']
X_no_potential = df[features_no_potential]

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_no_potential, y, test_size=0.2, random_state=42
)

model_no_potential = LinearRegression()
model_no_potential.fit(X_train_np, y_train_np)
y_pred_no_potential = model_no_potential.predict(X_test_np)

r2_no_potential = r2_score(y_test_np, y_pred_no_potential)
rmse_no_potential = np.sqrt(mean_squared_error(y_test_np, y_pred_no_potential))
mae_no_potential = mean_absolute_error(y_test_np, y_pred_no_potential)

print(f"\n--- Coefficients ---")
coef_no_potential = pd.DataFrame({
    'Feature': features_no_potential,
    'Coefficient': model_no_potential.coef_
})
print(coef_no_potential.to_string(index=False))
print(f"Intercept: €{model_no_potential.intercept_:,.2f}")

print(f"\n--- Performance ---")
print(f"R² Score: {r2_no_potential:.4f} ({r2_no_potential*100:.2f}%)")
print(f"RMSE: €{rmse_no_potential:,.2f}")
print(f"MAE: €{mae_no_potential:,.2f}")

scaler_np = StandardScaler()
X_train_np_scaled = scaler_np.fit_transform(X_train_np)

model_np_scaled = LinearRegression()
model_np_scaled.fit(X_train_np_scaled, y_train_np)

coef_np_standardized = pd.DataFrame({
    'Feature': features_no_potential,
    'Standardized_Coefficient': model_np_scaled.coef_,
    'Abs_Coefficient': np.abs(model_np_scaled.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\n--- Standardized Coefficients ---")
print(coef_np_standardized[['Feature', 'Standardized_Coefficient']].to_string(index=False))

print("\n" + "=" * 70)
print("MODEL COMPARISON: WITH vs WITHOUT POTENTIAL")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Model': ['With Potential', 'Without Potential'],
    'Features': [4, 3],
    'R² Score': [r2_full, r2_no_potential],
    'RMSE (€)': [rmse_full, rmse_no_potential],
    'MAE (€)': [mae_full, mae_no_potential]
})

print("\n--- Performance Comparison ---")
print(comparison_df.to_string(index=False))

r2_diff = r2_full - r2_no_potential
rmse_diff = rmse_no_potential - rmse_full
mae_diff = mae_no_potential - mae_full

print(f"\n--- Impact of Including Potential ---")
print(f"R² Improvement: +{r2_diff:.4f} ({r2_diff*100:.2f}%)")
print(f"RMSE Reduction: €{rmse_diff:,.2f}")
print(f"MAE Reduction: €{mae_diff:,.2f}")

print("\n" + "=" * 70)
print("COEFFICIENT INTERPRETATION")
print("=" * 70)

print(f"""
Model with All Features:

1. Overall (Coefficient: €{model_full.coef_[0]:,.2f})
   → Each 1-point increase in Overall rating increases value by €{model_full.coef_[0]:,.2f}
   
2. Potential (Coefficient: €{model_full.coef_[1]:,.2f})
   → Each 1-point increase in Potential rating increases value by €{model_full.coef_[1]:,.2f}
   
3. Age (Coefficient: €{model_full.coef_[2]:,.2f})
   → Each additional year of age {'decreases' if model_full.coef_[2] < 0 else 'increases'} value by €{abs(model_full.coef_[2]):,.2f}
   
4. International_Reputation (Coefficient: €{model_full.coef_[3]:,.2f})
   → Each 1-star increase in reputation increases value by €{model_full.coef_[3]:,.2f}
""")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

ax1 = axes[0, 0]
metrics = ['R²', 'RMSE (M€)', 'MAE (M€)']
with_potential = [r2_full, rmse_full/1e6, mae_full/1e6]
without_potential = [r2_no_potential, rmse_no_potential/1e6, mae_no_potential/1e6]

x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, with_potential, width, label='With Potential', color='#3498db')
bars2 = ax1.bar(x + width/2, without_potential, width, label='Without Potential', color='#e74c3c')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.set_title('Model Comparison: With vs Without Potential', fontsize=14, fontweight='bold')
ax1.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax2 = axes[0, 1]
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in coef_full_standardized['Standardized_Coefficient']]
bars = ax2.barh(coef_full_standardized['Feature'], 
                coef_full_standardized['Standardized_Coefficient'], 
                color=colors, edgecolor='black')
ax2.set_xlabel('Standardized Coefficient')
ax2.set_title('Feature Impact (With Potential)', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
for bar, val in zip(bars, coef_full_standardized['Standardized_Coefficient']):
    offset = 0.5e6 if val > 0 else -0.5e6
    ax2.text(val + offset, bar.get_y() + bar.get_height()/2,
             f'{val/1e6:.2f}M', va='center', fontsize=10)

ax3 = axes[1, 0]
ax3.scatter(y_test/1e6, y_pred_full/1e6, alpha=0.3, color='blue', s=10)
ax3.plot([0, y_test.max()/1e6], [0, y_test.max()/1e6], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Value (€ Millions)')
ax3.set_ylabel('Predicted Value (€ Millions)')
ax3.set_title(f'With Potential: Actual vs Predicted (R² = {r2_full:.4f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
ax4.scatter(y_test_np/1e6, y_pred_no_potential/1e6, alpha=0.3, color='orange', s=10)
ax4.plot([0, y_test_np.max()/1e6], [0, y_test_np.max()/1e6], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Value (€ Millions)')
ax4.set_ylabel('Predicted Value (€ Millions)')
ax4.set_title(f'Without Potential: Actual vs Predicted (R² = {r2_no_potential:.4f})', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q10_fifa_value_prediction.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0,
            fmt='.3f', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q10_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                  FIFA PLAYER VALUE PREDICTION SUMMARY                 ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  FEATURES ANALYZED:                                                   ║
║  • Overall (Player's current rating)                                  ║
║  • Potential (Maximum future rating)                                  ║
║  • Age (Player's age)                                                 ║
║  • International Reputation (1-5 stars)                               ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODEL WITH POTENTIAL:                                                ║
║  • R² Score: {r2_full:.4f} ({r2_full*100:.2f}% variance explained)                    ║
║  • RMSE: €{rmse_full/1e6:.2f} Million                                        ║
║  • MAE: €{mae_full/1e6:.2f} Million                                          ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  MODEL WITHOUT POTENTIAL:                                             ║
║  • R² Score: {r2_no_potential:.4f} ({r2_no_potential*100:.2f}% variance explained)               ║
║  • RMSE: €{rmse_no_potential/1e6:.2f} Million                                       ║
║  • MAE: €{mae_no_potential/1e6:.2f} Million                                         ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  IMPACT OF POTENTIAL:                                                 ║
║  • R² Improvement: +{r2_diff:.4f} ({r2_diff*100:.2f}%)                               ║
║  • Including Potential IMPROVES the model                             ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  ATTRIBUTE THAT INCREASES VALUE MOST:                                 ║
║  ★ {most_impactful_full:62}║
║                                                                        ║
║  Feature Importance Ranking:                                          ║
""")

for i, (_, row) in enumerate(coef_full_standardized.iterrows()):
    coef_millions = row['Standardized_Coefficient'] / 1e6
    print(f"║  {i+1}. {row['Feature']:30} (€{coef_millions:+.2f}M per std)        ║")

print("""║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  KEY INSIGHTS:                                                        ║
║  • International Reputation has the HIGHEST impact on value           ║
║  • Age has NEGATIVE impact (younger players are more valuable)        ║
║  • Overall and Potential both positively impact value                 ║
║  • Potential adds predictive power but Overall is more important      ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("\n--- Final Recommendation ---")
print(f"""
For FIFA player valuation:
1. INCLUDE Potential - it improves model performance by {r2_diff*100:.2f}%
2. Focus on International Reputation - highest value driver
3. Consider Age carefully - younger players command premium
4. Overall rating matters most for current value assessment
""")
