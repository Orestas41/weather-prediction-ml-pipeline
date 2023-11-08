"""
Evaluating previous tour results against models predictions.
Batch predicting next tour match results
"""
# pylint: disable=E0401, W0621, R0914, E1101, C0200, C0103, W0106, R0915
"""import csv
import os.path
import json
import pickle
import pandas as pd
import numpy as np"""
import mlflow
import logging
from datetime import datetime
import wandb
import argparse
import joblib
from datetime import datetime, timedelta
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Evaluating previous tour results against models predictions.
    Batch predicting next tour match results
    """
    LOGGER.info("7 - Running tour evaluation and prediction step")

    run = wandb.init(
        job_type="data_scraping")
    run.config.update(ARGS)

    model_local_path = run.use_artifact(ARGS.mlflow_model).download()

    df = pd.read_csv('../data/training_data.csv')

    predicted_data = pd.read_csv('../reports/next_week_prediction.csv')

    start = predicted_data.iloc[0]['time']
    end = predicted_data.iloc[-1]['time']

    mask = (df['time'] >= start) & (df['time'] <= end)
    recorded_data = df.loc[mask]

    predicted_data.rename(columns={'weathercode':'predicted_weathercode','temperature_2m_max':'predicted_temperature_2m_max','temperature_2m_min':'predicted_temperature_2m_min','precipitation_sum':'predicted_precipitation_sum'}
    , inplace=True)

    performance = pd.merge(recorded_data, predicted_data, on='time', how='outer')

    LOGGER.info("Calculating prediction error")
    performance['weathercode_performace'] = abs(
        performance['weathercode'] - performance['predicted_weathercode'])

    performance['max_temp_performace'] = abs(
        performance['temperature_2m_max'] - performance['predicted_temperature_2m_max'])

    performance['min_temp_performace'] = abs(
        performance['temperature_2m_min'] - performance['predicted_temperature_2m_min'])

    performance['precipitation_performace'] = abs(
        performance['precipitation_sum'] - performance['predicted_precipitation_sum'])

    LOGGER.info("Saving the report on the latest tour prediction evaluations")
    performance.to_csv(
        f"../reports/weekly_batch_prediction_performace.csv",
        index=None)

    ############################################################################

    with open('../config.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    # Create a date range for the next 7 days
    date_rng = pd.date_range(start=datetime.now(), end=datetime.now() + timedelta(days=7), freq='D', normalize=True)

    # Create a DataFrame with a date column
    df = pd.DataFrame(date_rng, columns=['time'])
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
    df['month-day'] = df['time'].dt.strftime('%m-%d')
    df['month-day'] = pd.to_datetime(df['month-day'], format='%m-%d')
    df['month-day'] = pd.to_datetime(df['month-day']).dt.strftime('%m%d').astype(int)
    df.set_index('time', inplace=True)
    df['city'] = config['cities']['Bristol']['id']

    #model = joblib.load("../training_validation/model_dir/model.joblib")
    model = mlflow.sklearn.load_model(model_local_path)

    preds = model.predict(df)

    df['predicted_weathercode'] = 0
    df['predicted_temperature_2m_max'] = 0
    df['predicted_temperature_2m_min'] = 0
    df['predicted_precipitation_sum'] = 0

    for i in range(len(preds)):
        df['predicted_weathercode'][i] = preds[i][0]
        df['predicted_temperature_2m_max'][i] = preds[i][1]
        df['predicted_temperature_2m_min'][i] = preds[i][2]
        df['predicted_precipitation_sum'][i] = preds[i][3]

    df.to_csv("../reports/next_week_prediction.csv")

    LOGGER.info("Batch tour evaluations and predictions finished")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    ARGS = PARSER.parse_args()

    go(ARGS)