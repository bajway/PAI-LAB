import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_samples = 1460

overall_qual = np.random.randint(1, 11, n_samples)
gr_liv_area = np.random.normal(1500, 500, n_samples).clip(400, 5000)
garage_cars = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.05, 0.15, 0.45, 0.30, 0.05])
year_built = np.random.randint(1900, 2011, n_samples)

sale_price = (
    overall_qual * 15000 +
    gr_liv_area * 50 +
    garage_cars * 12000 +
    (year_built - 1900) * 500 +
    np.random.normal(0, 20000, n_samples)
).clip(50000, 800000)

df = pd.DataFrame({
    'OverallQual': overall_qual,
    'GrLivArea': gr_liv_area,
    'GarageCars': garage_cars,
    'YearBuilt': year_built,
    'SalePrice': sale_price
})

print("=" * 70)
print("Q8: HOUSE PRICE PREDICTION - MULTIPLE LINEAR REGRESSION")
print("=" * 70)

print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print(df.head(10))

print("\n--- Statistical Summary ---")
print(df.describe().round(2))

print("\n--- Correlation Matrix ---")
correlation_matrix = df.corr()
print(correlation_matrix.round(4))

features = ['OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n--- Train/Test Split ---")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n" + "=" * 70)
print("MODEL COEFFICIENTS")
print("=" * 70)

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_,
    'Abs_Coefficient': np.abs(model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print(f"\nIntercept: ${model.intercept_:,.2f}")
print("\n--- Feature Coefficients ---")
print(coefficients[['Feature', 'Coefficient']].to_string(index=False))

print("\n" + "=" * 70)
print("COEFFICIENT INTERPRETATION")
print("=" * 70)

print(f"""
1. OverallQual (Coefficient: ${model.coef_[0]:,.2f})
   → For each 1-point increase in overall quality (1-10 scale),
     the house price increases by ${model.coef_[0]:,.2f}
   
2. GrLivArea (Coefficient: ${model.coef_[1]:,.2f})
   → For each additional square foot of living area,
     the house price increases by ${model.coef_[1]:,.2f}
   
3. GarageCars (Coefficient: ${model.coef_[2]:,.2f})
   → For each additional car the garage can hold,
     the house price increases by ${model.coef_[2]:,.2f}
   
4. YearBuilt (Coefficient: ${model.coef_[3]:,.2f})
   → For each year newer the house is,
     the house price increases by ${model.coef_[3]:,.2f}
""")

print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\nR² Score: {r2:.4f} ({r2*100:.2f}%)")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"MAPE: {mape:.2f}%")

print(f"\nInterpretation:")
print(f"  - R² = {r2:.4f} means {r2*100:.2f}% of the variance in house prices")
print(f"    is explained by the model")
print(f"  - RMSE = ${rmse:,.2f} means on average, predictions are off by this amount")

print("\n" + "=" * 70)
print("FEATURE IMPACT ANALYSIS")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_scaled = LinearRegression()
model_scaled.fit(X_train_scaled, y_train)

standardized_coef = pd.DataFrame({
    'Feature': features,
    'Standardized_Coefficient': model_scaled.coef_,
    'Abs_Std_Coefficient': np.abs(model_scaled.coef_)
}).sort_values('Abs_Std_Coefficient', ascending=False)

print("\n--- Standardized Coefficients (Feature Impact) ---")
print(standardized_coef[['Feature', 'Standardized_Coefficient']].to_string(index=False))

most_impactful = standardized_coef.iloc[0]['Feature']
print(f"\n★ MOST IMPACTFUL FEATURE: {most_impactful}")
print(f"  (Highest standardized coefficient magnitude)")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

ax1 = axes[0, 0]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
bars = ax1.bar(coefficients['Feature'], coefficients['Coefficient'], color=colors, edgecolor='black')
ax1.set_xlabel('Feature', fontsize=12)
ax1.set_ylabel('Coefficient ($)', fontsize=12)
ax1.set_title('Raw Coefficients', fontsize=14, fontweight='bold')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'${height:,.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

ax2 = axes[0, 1]
bars2 = ax2.barh(standardized_coef['Feature'], standardized_coef['Standardized_Coefficient'], 
                  color=colors, edgecolor='black')
ax2.set_xlabel('Standardized Coefficient', fontsize=12)
ax2.set_title('Feature Impact (Standardized)', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred, alpha=0.5, color='blue', edgecolors='black')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Price ($)', fontsize=12)
ax3.set_ylabel('Predicted Price ($)', fontsize=12)
ax3.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = axes[1, 1]
residuals = y_test - y_pred
ax4.scatter(y_pred, residuals, alpha=0.5, color='green', edgecolors='black')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Price ($)', fontsize=12)
ax4.set_ylabel('Residuals ($)', fontsize=12)
ax4.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q8_house_price_regression.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('q8_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary_df = pd.DataFrame({
    'Metric': ['R² Score', 'RMSE', 'MAE', 'MAPE', 'Intercept', 'Most Impactful Feature'],
    'Value': [f'{r2:.4f}', f'${rmse:,.2f}', f'${mae:,.2f}', f'{mape:.2f}%', 
              f'${model.intercept_:,.2f}', most_impactful]
})

print(summary_df.to_string(index=False))

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                         KEY FINDINGS                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Model Equation:                                                       ║
║  SalePrice = {model.intercept_:,.0f}                                   ║
║              + {model.coef_[0]:,.0f} × OverallQual                     ║
║              + {model.coef_[1]:,.0f} × GrLivArea                       ║
║              + {model.coef_[2]:,.0f} × GarageCars                      ║
║              + {model.coef_[3]:,.0f} × YearBuilt                       ║
║                                                                        ║
║  Most Impactful Feature: {most_impactful:45}║
║                                                                        ║
║  Model explains {r2*100:.1f}% of house price variance                  ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
