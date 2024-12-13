# Performs cross-validation to validate the consistency and
# reliability of the model using training data.

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

X_train, X_test, y_train, y_test = joblib.load('train_test_split.pkl')
model = LinearRegression()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

# Convert negative scores to positive
cv_scores = -cv_scores

print("Cross-validation scores (MAE):", cv_scores)
print(f"Mean MAE: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of MAE: {np.std(cv_scores):.4f}")
