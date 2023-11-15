"""
Evaluating previous tour results against models predictions.
Batch predicting next tour match results
"""
# pylint: disable=E0401, W0621, R0914, E1101, C0200, C0103, W0106, R0915
import mlflow
import logging
from datetime import datetime
import wandb
import os
import tempfile
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Set up logging
logging.basicConfig(
    filename=f"../{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Evaluating previous tour results against models predictions.
    Batch predicting next tour match results
    """
    LOGGER.info("7 - Running weekly batch prediction and evaluation step")

    run = wandb.init(project="weather-prediction",
        job_type="batch_prediction")
    run.config.update(ARGS)

    LOGGER.info(
        "Downloading models- %s, %s and data- %s artifacts",
        ARGS.reg_model,
        ARGS.class_model,
        ARGS.full_dataset,
        ARGS.batch_prediction
    )
    # Downloading model artifact
    reg_model_local_path = run.use_artifact(ARGS.reg_model).download()
    class_model_local_path = run.use_artifact(ARGS.class_model).download()
    # Downloading test dataset
    full_dataset_path = run.use_artifact(ARGS.full_dataset).file()
    batch_prediction_path = run.use_artifact(ARGS.batch_prediction).file()
    
    latest_data = pd.read_csv(full_dataset_path)

    LOGGER.info("Opening last weeks predictions")
    predicted_data = pd.read_csv(batch_prediction_path)

    LOGGER.info("Locating prediction range")
    # Last weeks prediction range
    start = predicted_data.iloc[0]['time']
    end = predicted_data.iloc[-1]['time']

    # Finding recorded data from the range that was predicted last week
    mask = (latest_data['time'] >= start) & (latest_data['time'] <= end)
    recorded_data = latest_data.loc[mask]

    # Preparing features
    predicted_data.rename(columns={'weathercode':'predicted_weathercode','temperature_2m_max':'predicted_temperature_2m_max','temperature_2m_min':'predicted_temperature_2m_min','precipitation_sum':'predicted_precipitation_sum'}
    , inplace=True)

    LOGGER.info("Merging predicted and recorded weather data")
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
    
    # Dripping irrelevant columns
    performance = performance.drop(['month-day_x','city_x','month-day_y','city_y'], axis=1)

    LOGGER.info("Average Weathercode performace: %s", performance['weathercode_performace'].mean())
    LOGGER.info("Average Max temperature performace: %s", performance['max_temp_performace'].mean())
    LOGGER.info("Average Min temperature performace: %s", performance['min_temp_performace'].mean())
    LOGGER.info("Average Precipitation performace: %s", performance['precipitation_performace'].mean())
    
    LOGGER.info("Setting up prediction for the next week")

    # Create a date range for the next 7 days
    date_rng = pd.date_range(start=datetime.now(), end=datetime.now() + timedelta(days=7), freq='D', normalize=True)

    # Create a DataFrame with a date column
    prediction = pd.DataFrame(date_rng, columns=['time'])
    prediction['time'] = pd.to_datetime(prediction['time'], format='%Y-%m-%d')
    prediction['month-day'] = prediction['time'].dt.strftime('%m-%d')
    prediction['month-day'] = pd.to_datetime(prediction['month-day'], format='%m-%d')
    prediction['month-day'] = pd.to_datetime(prediction['month-day']).dt.strftime('%m%d').astype(int)
    prediction.set_index('time', inplace=True)
    prediction['city'] = predicted_data['city'][0]

    LOGGER.info("Inference")
    reg_model = mlflow.sklearn.load_model(reg_model_local_path)
    class_model = mlflow.sklearn.load_model(class_model_local_path)
    reg_pred = reg_model.predict(prediction)

    prediction['temperature_2m_max'] = 0
    prediction['temperature_2m_min'] = 0
    prediction['precipitation_sum'] = 0

    for i in range(len(reg_pred)):
        prediction['temperature_2m_max'][i] = reg_pred[i][0]
        prediction['temperature_2m_min'][i] = reg_pred[i][1]
        prediction['precipitation_sum'][i] = reg_pred[i][2]

    prediction = prediction[['temperature_2m_max','temperature_2m_min','precipitation_sum','month-day','city']]

    class_pred = class_model.predict(prediction)

    prediction['weathercode'] = 0
    
    for i in range(len(class_pred)):
        prediction['weathercode'][i] = class_pred[i]


    prediction = prediction.rename(columns = {'temperature_2m_max':'predicted_temperature_2m_max',
                                                'weathercode':'predicted_weathercode',
                                                'temperature_2m_min':'predicted_temperature_2m_min',
                                                'precipitation_sum':'predicted_precipitation_sum',})

    LOGGER.info("Average Weathercode prediction: %s", prediction['predicted_weathercode'].mean())
    LOGGER.info("Average Max temperature prediction: %s", prediction['predicted_temperature_2m_max'].mean())
    LOGGER.info("Average Min temperature prediction: %s", prediction['predicted_temperature_2m_min'].mean())
    LOGGER.info("Average Precipitation prediction: %s", prediction['predicted_precipitation_sum'].mean())

    LOGGER.info("Plotting next weeks weather prediction")
    temperature_range = (prediction['predicted_temperature_2m_max'] + prediction['predicted_temperature_2m_min']) / 2
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.fill_between(prediction.index, prediction['predicted_temperature_2m_min'], prediction['predicted_temperature_2m_max'], alpha=0.5)
    ax.plot(temperature_range, label='Mean temperature', marker='o', color='blue')
    ax.plot(prediction['predicted_temperature_2m_max'], label='Max temperature', color='red', marker='s')
    ax.plot(prediction['predicted_temperature_2m_min'], label='Min temperature', color='blue', marker='^')
    ax2.fill_between(prediction.index, prediction['predicted_precipitation_sum'], 0, alpha=0.5, color='cyan')
    ax2.plot(prediction['predicted_precipitation_sum'], label='Precipitation amount', color='cyan', marker=',')
    for x, y in zip(prediction.index, temperature_range):
        plt.annotate(str(round(y,1)), xy=(x, y), ha='center', va='center') 
    for x, y in zip(prediction.index, prediction['predicted_precipitation_sum']):
        plt.annotate(str(round(y,1)), xy=(x, y), ha='center', va='center')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Weather predictions for Bristol')
    ax.set_ylim(0, 22)
    ax2.set_ylabel('Precipitation (mm)')
    ax2.set_ylim(0, 30)
    ax.grid(True)
    ax.legend()

    fig = wandb.Image(plt)
    run.log(({"plot": fig}))

    for file_name, k, desc in zip([performance, prediction],
                                  ['batch_prediction_performance.csv', 'batch_prediction.csv'],
                                  ['batch_prediction_performance','batch_prediction']):
        LOGGER.info("Uploading %s", desc)
        with tempfile.NamedTemporaryFile("w") as file:
            file_name.to_csv(file.name, index=True)
            artifact = wandb.Artifact(
                k,
                type=desc,
                description=desc,
            )
            artifact.add_file(file.name)
            run.log_artifact(artifact)
            if not os.getenv('TESTING'):
                artifact.wait()
            else:
                pass

    LOGGER.info("Batch tour evaluations and predictions finished")

    log_file = f"../{datetime.now().strftime('%Y-%m-%d')}.log"
    log_artifact = wandb.Artifact(
        "log_file",
        type="log",
        description="Log file",
    )
    log_artifact.add_file(log_file)
    run.log_artifact(log_artifact)
    
    os.remove(f"../{datetime.now().strftime('%Y-%m-%d')}.log")

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument(
        "--reg_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    PARSER.add_argument(
        "--class_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    PARSER.add_argument(
        "--full_dataset",
        type=str,
        help="Full dataset",
        required=True
    )

    PARSER.add_argument("--batch_prediction", type=str, help="Input artifact to split", required=True)

    ARGS = PARSER.parse_args()

    go(ARGS)