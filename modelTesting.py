# Evaluates the model’s performance on the test dataset using metrics like MAE, RMSE, and R²

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

X_train, X_test, y_train, y_test = joblib.load('train_test_split.pkl')
model = joblib.load('trained_model.pkl')
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Results
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
