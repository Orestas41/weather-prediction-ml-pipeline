"""
This script run W&B sweep agent that retrains regression model with parameters
in search for best performing configuration. This script is not in the pipeline
"""
import os
import wandb
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model():
    """
    Runs W&B sweep to find best model configuration
    """
    # Setting W&B project and group
    os.environ["WANDB_PROJECT"] = "weather-prediction"
    os.environ["WANDB_RUN_GROUP"] = 'regression model tuning'

    # Initiating W&B run
    run = wandb.init(config=sweep_config)

    # Opening training data
    df = pd.read_csv('data/training_data.csv')

    # Setting time column as index
    df.set_index('time', inplace=True)

    # Setting input and target features
    X = df.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
    y = df[['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

    # Splitting data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3)

    # Setting up the configuration
    config = run.config

    # Specifying parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': config.n_estimators,
        'learning_rate': config.learning_rate,
        'max_depth': config.max_depth,
        'min_child_weight': config.min_child_weight,
        'gamma': config.gamma,
        'reg_alpha': config.reg_alpha,
        'reg_lambda': config.reg_lambda,
        'subsample': config.subsample,
        'colsample_bytree': config.colsample_bytree
    }

    # Preparing the regression model
    model = XGBRegressor(**params)

    # Fitting
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=config.early_stopping_rounds, verbose=False)
    
    # Predicting
    y_pred = model.predict(X_val)

    # Scoring
    r_squared = model.score(X_val, y_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val,y_pred))

    # Logging scores to W&B
    wandb.log({'r_squared':r_squared})
    wandb.log({'mae':mae})
    wandb.log({'rmse':rmse})


# Setting up parameter limits
sweep_config = {
    'method':'random',
    'metric': {
        'name':'rmse',
        'goal':'minimize'
    },
    'parameters':{
        'learning_rate':{
            'min': 0.01,
            'max': 0.1
        },
        'max_depth':{
            'values':[3,5,7,9]
        },
        'subsample':{
            'min': 0.5,
            'max':1.0
        },
        'colsample_bytree':{
            'min':0.5,
            'max':1.0
        },
        'n_estimators':{
            'min':100,
            'max':500
        },
        'early_stopping_rounds':{
            'values': [10,20,30]
        },
        'min_child_weight':{
            'min':1,
            'max':5
        },
        'gamma':{
            'min':0.0,
            'max':0.5
        },
        'reg_alpha':{
            'min':0.0,
            'max':0.5
        },
        'reg_lambda':{
            'min':0.0,
            'max':0.5
        }}}
    
# Setting up W&B sweeps
sweep_id = wandb.sweep(sweep_config,project="weather-prediction")

# Initiating W&B agent
wandb.agent(sweep_id, function=train_model)