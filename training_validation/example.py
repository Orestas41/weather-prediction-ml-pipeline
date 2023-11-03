"""
This script trains and validates the model
"""
# pylint: disable=E0401, C0103, R0914, E1101, W0621
import argparse
import logging
import os
import shutil
import json
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import mlflow
import wandb

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
        project="project-FootballPredict",
        job_type="training_validation")
    run.config.update(ARGS)

    # Getting the Linear Regression configuration and updating W&B
    with open(ARGS.model_config) as file:
        model_config = json.load(file)
    run.config.update(model_config)

    LOGGER.info(
        "Fetching %s and setting it as dataframe", ARGS.trainval_artifact)
    trainval_local_path = run.use_artifact(ARGS.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)

    LOGGER.info("Setting winner column as target")
    y = X.pop('Winner')

    LOGGER.info("Number of outcomes: %s", y.nunique())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=ARGS.val_size)

    LOGGER.info("Preparing Linear Regression model")
    model = LinearRegression(**model_config)

    # Fitting it to the X_train, y_train data
    LOGGER.info("Fitting")
    model.fit(X_train, y_train)

    # Evaluating the model
    LOGGER.info("Scoring the model")
    r_squared = model.score(X_val, y_val)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    LOGGER.info("Score: %s", r_squared)
    LOGGER.info("MAE: %s", mae)

    LOGGER.info("Exporting model")
    if os.path.exists("model_dir"):
        shutil.rmtree("model_dir")

    mlflow.sklearn.save_model(model, "model_dir")

    # Uploading inference pipeline artifact to W&B
    LOGGER.info("Saving and exporting the model")
    artifact = wandb.Artifact(
        ARGS.output_artifact,
        type='model_export',
        description='model pipeline',
        metadata=model_config
    )
    artifact.add_dir("model_dir")
    run.log_artifact(artifact)

    # Saving r_squared as a summary
    run.summary['r2'] = r_squared

    # Logging the variable "mae" as a summary
    run.summary['mae'] = mae

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
        "--model_config",
        help="Model configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for Linear Regression.",
        default="{}",
    )

    PARSER.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    ARGS = PARSER.parse_args()

    go(ARGS)