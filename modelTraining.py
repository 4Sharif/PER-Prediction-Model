# Trains a Linear Regression model on the training data and
# saves the trained model for evaluation.

from sklearn.linear_model import LinearRegression
import joblib

X_train, X_test, y_train, y_test = joblib.load('train_test_split.pkl') # Load the split
model = LinearRegression() # Initialize the model
model.fit(X_train, y_train) # Train the model on the training data

joblib.dump(model, 'trained_model.pkl')

# Output model coefficients and intercept
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
