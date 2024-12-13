# Compares the performance of Linear Regression, Random Forest,
# Gradient Boosting, and Support Vector Regressor models.

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
import joblib
from split import X_train, X_test, y_train, y_test
from modelTraining import model

models = { "Linear Regression": model, "Random Forest": RandomForestRegressor(random_state=42), "Gradient Boosting": GradientBoostingRegressor(random_state=42), "Support Vector Regressor": SVR(kernel="rbf") }

results = {}

for name, regressor in models.items():
    if name != "Linear Regression":
        regressor.fit(X_train, y_train)
        joblib.dump(regressor, filename=f"{name.replace(' ', '_')}.joblib")

    y_pred = regressor.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R²": r2}

results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ["Model", "MAE", "RMSE", "R²"]

# Results
print("Model Comparison Results:")
print(results_df)
results_df.to_csv("model_comparison_results.csv", index=False)

