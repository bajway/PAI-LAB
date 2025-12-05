import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7],
    'Salary': [39.0, 46.0, 47.0, 52.0, 56.0, 64.0, 65.0, 67.0, 68.0, 70.0]
}

df = pd.DataFrame(data)

X = df[['YearsExperience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept: {intercept:.4f}")
print(f"Slope: {slope:.4f}")

prediction_4_5 = model.predict([[4.5]])[0]
print(f"Predicted Salary for 4.5 years of experience: {prediction_4_5:.4f} thousands")

r2 = model.score(X, y)
print(f"R² Value: {r2:.4f}")

print(f"\nInterpretation: {r2*100:.2f}% of the variance in Salary is explained by YearsExperience")
