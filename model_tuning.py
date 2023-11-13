import wandb
import pandas as pd
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

"""run = wandb.init(
    project="weather-prediction",
    job_type="model-tuning")"""

def train_model():

    df = pd.read_csv('data/training_data.csv')

    X = df.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
    y = df[['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3)

    X_train.set_index('time', inplace=True)
    X_val.set_index('time', inplace=True)   

    run = wandb.init(config=sweep_config,project="weather-prediction",
    job_type="model-tuning")

    config = run.config

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

    model = XGBRegressor(**params)



    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=config.early_stopping_rounds, verbose=False)
    
    y_pred = model.predict(X_val)

    r_squared = model.score(X_val, y_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val,y_pred))

    wandb.log({'r_squared':r_squared})
    wandb.log({'mae':mae})
    wandb.log({'rmse':rmse})

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
    
sweep_id = wandb.sweep(sweep_config,project="weather-prediction")

wandb.agent(sweep_id, function=train_model)