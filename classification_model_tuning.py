import wandb
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model():

    os.environ["WANDB_PROJECT"] = "weather-prediction"
    os.environ["WANDB_RUN_GROUP"] = 'classification model tuning'

    df = pd.read_csv('data/training_data.csv')

    X = df.drop(['weathercode'], axis=1)
    y = df[['weathercode']]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3)

    X_train.set_index('time', inplace=True)
    X_val.set_index('time', inplace=True)   

    run = wandb.init(config=sweep_config)

    config = run.config

    params = {
        'n_estimators': config.n_estimators,
        'max_depth': config.max_depth,
        "min_samples_split": config.min_samples_split,
        "min_samples_leaf": config.min_samples_leaf,
        "criterion": config.criterion,
        "max_features": config.max_features,
        "bootstrap": config.bootstrap
    }

    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)
    
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
        'max_depth':{
            'values':[None,3,5,7,9]
        },
        'n_estimators':{
            'min':100,
            'max':500
        },
        'min_samples_split':{
            'values': [2,5,10]
        },
        'min_samples_leaf':{
            'min':1,
            'max':5
        },
        'criterion':{
            'values':["gini", "entropy", "log_loss"]
        },
        'max_features':{
            'values':[None, "sqrt", "log2"]
        },
        'bootstrap':{
            'values':[True, False]
        }}}
    
sweep_id = wandb.sweep(sweep_config,project="weather-prediction")

wandb.agent(sweep_id, function=train_model)