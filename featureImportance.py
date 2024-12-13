# Identifies and ranks the importance of features based on the
# Linear Regression model coefficients.

import pandas as pd
import joblib

model = joblib.load('trained_model.pkl')

feature_names = ['Total_Minutes', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PTS', 'PTOV', 'SFD', 'PGA', 'AND1', 'TS%', 'USG%', 'WS', 'BPM', 'VORP', 'ORtg']
coefficients = model.coef_
absolute_importance = abs(coefficients)

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute Importance': absolute_importance
})

# Sorting by absolute importance
feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)  

print("Feature Importance for Linear Regression:")
print(feature_importance)

