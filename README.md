# Player Efficiency Prediction Model (PER)

This project applies data mining and machine learning to predict Player Efficiency Rating (PER) using advanced NBA statistics. Built using Python, the model draws from 11 seasons of NBA data (2014–2024) and was designed with a focus on predictive accuracy, feature analysis, and real-world usability.

## Overview

- **Goal**: Predict a player's PER based on 18 statistical features
- **Method**: Linear Regression (with cross-validation and model comparison)
- **Dataset**: 5,792 NBA player-seasons, filtered to 4,490 after preprocessing
- **Outcome**: Reliable model (R² = 0.9542) with a real-time prediction tool

## File Execution Order

The code is meant to be run in the following order:

1. `split.py` – Prepares the normalized dataset
2. `modelTraining.py` – Trains the Linear Regression model
3. `crossValidation.py` – Evaluates performance using 5-fold validation
4. `featureImportance.py` – Displays most influential stats
5. `modelTesting.py` – Tests the model on holdout data
6. `modelComparison.py` – Compares performance of multiple models
7. `predictiveTool.py` – Interactive prediction tool using player inputs

## Model Performance

- **R²**: 0.9542
- **MAE**: 0.0283
- **RMSE**: 0.0363
- **Avg Difference**: ±1.81 PER points (based on test players)

## Key Features Used

- Minutes Played (MP), Field Goals (FG, FGA), Free Throws (FT, FTA)  
- Rebounds (TRB), Assists (AST), Points (PTS)  
- Turnovers (PTOV), Fouls Drawn (SFD), Assists Generated (PGA), And1s  
- Advanced metrics like TS%, USG%, WS, BPM, VORP, ORtg  

All features are scaled using min-max normalization for model compatibility.

## Dependencies

- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

## How to Use

1. Download the transformed CSV dataset  
2. Copy its file path and paste it into `split.py`  
3. Run `split.py` and `modelTraining.py` first (this step saves the model)  
4. Then run `predictiveTool.py` to generate PER predictions from user input

The predictive tool includes a built-in test set of 28 players (1997–2024) to verify model accuracy.
