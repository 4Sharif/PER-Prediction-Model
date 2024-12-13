# Splits the dataset into training and testing subsets for
# model evaluation. The split ensures 80% of the data is used
# for training, and 20% is reserved for testing.
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Copy the file path for transformed dataset and insert it here
data = pd.read_csv('/Users/mohamed/Documents/CompSci/Data Mining/Project2/MinMax.csv')

X = data[['Total_Minutes', 'FG', 'FGA', 'FT', 'FTA', 'TRB', 'AST', 'PTS', 'PTOV', 'SFD', 'PGA', 'AND1', 'TS%', 'USG%', 'WS', 'BPM', 'VORP', 'ORtg']]
y = data['PER']

# 80% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
joblib.dump((X_train, X_test, y_train, y_test), 'train_test_split.pkl')

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
