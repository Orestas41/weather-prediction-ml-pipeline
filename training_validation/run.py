"""
This script trains and validates the models
"""
# pylint: disable=E0401, C0103, R0914, E1101, W0621
import logging
import json
from datetime import datetime
import mlflow
import wandb
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier

# Set up logging
logging.basicConfig(
    filename=f"../{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Train and validate the models
    """
    LOGGER.info("4 - Running training and validation step")

    run = wandb.init(job_type="training_validation")
    run.config.update(ARGS)

    LOGGER.info(
        "Fetching %s and setting it as dataframe", ARGS.trainval_artifact)
    trainval_local_path = run.use_artifact(ARGS.trainval_artifact).file()
    df = pd.read_csv(trainval_local_path)

    # Setting time column as the index
    df.set_index('time', inplace=True)

    LOGGER.info("Getting model configurations")
    # Getting the XGBRegressor configuration
    with open(ARGS.reg_config) as reg_file:
        reg_config = json.load(reg_file)

    # Getting the RandomForestClassifier configuration
    with open(ARGS.class_config) as class_file:
        class_config = json.load(class_file)

    LOGGER.info('Splitting data into training and validation')
    X_train, X_val, y_train, y_val = train_test_split(
        df, df, test_size=ARGS.val_size)

    # Setting up features for regression model
    input_features = ['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']
    reg_features = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']

    LOGGER.info('Setting up data for regression model')
    reg_X_train = X_train.drop(input_features, axis=1)
    reg_y_train = y_train[reg_features]
    reg_X_val = X_val.drop(input_features, axis=1)
    reg_y_val = y_val[reg_features]

    LOGGER.info('Setting up data for classification model')
    class_X_train = X_train.drop(['weathercode'], axis=1)
    class_y_train = y_train[['weathercode']]
    class_X_val = X_val.drop(['weathercode'], axis=1)
    class_y_val = y_val[['weathercode']]

    LOGGER.info("Preparing models with configuration")
    reg_model = XGBRegressor(**reg_config)
    class_model = RandomForestClassifier(**class_config)

    LOGGER.info("Fitting")
    reg_model.fit(reg_X_train, reg_y_train)
    class_model.fit(class_X_train, class_y_train)

    LOGGER.info("Predicting")
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

    LOGGER.info("Saving models")
    for path, model in zip(['../reg_model_dir','../class_model_dir'],
                           [reg_model, class_model]):
        mlflow.sklearn.save_model(model, path)

    LOGGER.info("Training and validation finished")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(description="Training and validation models")

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
        "--reg_config",
        help="Model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for XGBRegressor.",
        default="{}",
    )

    PARSER.add_argument(
        "--class_config",
        help="Model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for Random Forest Classifier.",
        default="{}",
    )

    ARGS = PARSER.parse_args()

    go(ARGS)