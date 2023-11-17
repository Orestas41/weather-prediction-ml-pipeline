"""
This script run W&B sweep agent that retrains classification model with parameters
in search for best performing configuration. This script is not in the pipeline
"""
# pylint: disable=C0103
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import wandb

def train_model():
    """
    Runs W&B sweep to find best model configuration
    """
    # Setting W&B project and group
    os.environ["WANDB_PROJECT"] = "weather-prediction"
    os.environ["WANDB_RUN_GROUP"] = 'classification model tuning'

    # Initiating W&B run
    run = wandb.init(config=sweep_config)

    # Opening training data
    df = pd.read_csv('data/training_data.csv')

    # Setting time column as index
    df.set_index('time', inplace=True)

    # Setting input and target features
    X = df.drop(['weathercode'], axis=1)
    y = df[['weathercode']]

    # Splitting data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3)

    # Setting up the configuration
    config = run.config

    # Specifying parameters
    params = {
        'n_estimators': config.n_estimators,
        'max_depth': config.max_depth,
        "min_samples_split": config.min_samples_split,
        "min_samples_leaf": config.min_samples_leaf,
        "criterion": config.criterion,
        "max_features": config.max_features,
        "bootstrap": config.bootstrap
    }

    # Preparing the classification model
    model = RandomForestClassifier(**params)

    # Fitting
    model.fit(X_train, y_train)

    # Predicting
    y_pred = model.predict(X_val)

    # Scoring
    r_squared = model.score(X_val, y_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Logging scores to W&B
    wandb.log({'r_squared': r_squared})
    wandb.log({'mae': mae})
    wandb.log({'rmse': rmse})


# Setting up parameter limits
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'
    },
    'parameters': {
        'max_depth': {
            'values': [None, 3, 5, 7, 9]
        },
        'n_estimators': {
            'min': 100,
            'max': 500
        },
        'min_samples_split': {
            'values': [2, 5, 10]
        },
        'min_samples_leaf': {
            'min': 1,
            'max': 5
        },
        'criterion': {
            'values': ["gini", "entropy", "log_loss"]
        },
        'max_features': {
            'values': [None, "sqrt", "log2"]
        },
        'bootstrap': {
            'values': [True, False]
        }}}

# Setting up W&B sweeps
sweep_id = wandb.sweep(sweep_config, project="weather-prediction")

# Initiating W&B agent
wandb.agent(sweep_id, function=train_model)
