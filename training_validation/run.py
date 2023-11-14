"""
This script trains and validates the model
"""
# pylint: disable=E0401, C0103, R0914, E1101, W0621

import logging
import shutil
import json
from datetime import datetime
import mlflow
import tempfile
import wandb
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Train and validation the model
    """
    LOGGER.info("5 - Running training and validation step")

    run = wandb.init(
        project="weather-prediction",
        job_type="training_validation")
    run.config.update(ARGS)

    LOGGER.info("Setting up file locations according to the environment")
    if not os.getenv('TESTING'):
        reg_model_path = 'reg_model_dir'
        class_model_path = "class_model_dir"
    else:
        # Use a temporary directory for testing
        if not os.path.exists('data'):
            os.makedirs('data')
        reg_model_path = os.path.join(tempfile.gettempdir(), 'reg_model_dir')
        class_model_path = os.path.join(tempfile.gettempdir(), "class_model_dir")

    # Getting the XGBRegressor configuration and updating W&B
    with open(ARGS.reg_model_config) as file:
        reg_model_config = json.load(file)
    run.config.update(reg_model_config)

    with open(ARGS.class_model_config) as file:
        class_model_config = json.load(file)
    run.config.update(class_model_config)

    LOGGER.info(
        "Fetching %s and setting it as dataframe", ARGS.trainval_artifact)
    trainval_local_path = run.use_artifact(ARGS.trainval_artifact).file()

    df = pd.read_csv(trainval_local_path)

    LOGGER.info("Setting feature and target columns")

    LOGGER.info('Splitting data into training and validation')
    X_train, X_val, y_train, y_val = train_test_split(
        df, df, test_size=ARGS.val_size)

    X_train.set_index('time', inplace=True)
    X_val.set_index('time', inplace=True)

    input_features = ['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']
    reg_features = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']

    reg_X_train = X_train.drop(input_features, axis=1)
    reg_y_train = y_train[reg_features]
    reg_X_val = X_val.drop(input_features, axis=1)
    reg_y_val = y_val[reg_features]

    class_X_train = X_train.drop(['weathercode'], axis=1)
    class_y_train = y_train[['weathercode']]
    class_X_val = X_val.drop(['weathercode'], axis=1)
    class_y_val = y_val[['weathercode']]

    LOGGER.info("Preparing XGBRegressor model")
    reg_model = XGBRegressor(**reg_model_config)
    class_model = RandomForestClassifier(**class_model_config)

    LOGGER.info("Fitting")
    reg_model.fit(reg_X_train, reg_y_train)
    class_model.fit(class_X_train, class_y_train)

    reg_y_pred = reg_model.predict(reg_X_val)
    class_y_pred = class_model.predict(class_X_val)

    LOGGER.info("Scoring the model")
    reg_r_squared = reg_model.score(reg_X_val, reg_y_val)
    reg_mae = mean_absolute_error(reg_y_val, reg_y_pred)

    class_r_squared = class_model.score(class_X_val, class_y_val)
    class_mae = mean_absolute_error(class_y_val, class_y_pred)

    LOGGER.info("Regression Score: %s", reg_r_squared)
    LOGGER.info("Regression MAE: %s", reg_mae)
    LOGGER.info("Classification Score: %s", class_r_squared)
    LOGGER.info("Classification MAE: %s", class_mae)

    LOGGER.info("Total Score: %s", (class_r_squared+reg_r_squared)/2)
    LOGGER.info("Total MAE: %s", (class_mae+reg_mae)/2)

    LOGGER.info("Exporting model")
    if os.path.exists(reg_model_path):
        shutil.rmtree(reg_model_path)

    if os.path.exists(class_model_path):
        shutil.rmtree(class_model_path)

    mlflow.sklearn.save_model(reg_model, reg_model_path)
    mlflow.sklearn.save_model(class_model, class_model_path)

    LOGGER.info("Exporting the model artifacts")
    for n, name, meta, r2, mae in zip(['reg', 'class'],
                                     ['Regression','Classification'],
                                     [reg_model_config, class_model_config],
                                     [reg_r_squared,class_r_squared],
                                     [reg_mae,class_mae]):
        artifact = wandb.Artifact(
            f'{n}_model',
            type='model_export',
            description=f'{name} model',
            metadata=meta
        )
        
        if not os.getenv('TESTING'):
            artifact.add_dir(f"{n}_model_dir")
            run.log_artifact(artifact)
            artifact.wait()
            # Saving r_squared as a summary
            run.summary[f'{n}_r2'] = r2

            # Logging the variable "mae" as a summary
            run.summary[f'{n}_mae'] = mae
        else:
            pass
        

    LOGGER.info("Finished training and validation")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Basic cleaning of dataset")

    PARSER.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    PARSER.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    PARSER.add_argument(
        "--reg_model_config",
        help="Model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for Linear Regression.",
        default="{}",
    )

    PARSER.add_argument(
        "--class_model_config",
        help="Model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for Linear Regression.",
        default="{}",
    )

    ARGS = PARSER.parse_args()

    go(ARGS)