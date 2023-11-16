"""
Evaluating previous weather prediction results against real world data.
Batch predicting the weather of last and next week
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
    Evaluating previous weather prediction results against real world data.
    Batch predicting the weather of last and next week
    """
    LOGGER.info("6 - Running batch prediction and evaluation step")

    # Initiating wandb run
    run = wandb.init(job_type="batch_prediction")
    run.config.update(ARGS)

    LOGGER.info(
        "Downloading %s, %s, %s and %s",
        ARGS.reg_model,
        ARGS.class_model,
        ARGS.full_dataset,
        ARGS.batch_prediction
    )

    # Downloading model artifacts
    reg_model_local_path = run.use_artifact(ARGS.reg_model).download()
    class_model_local_path = run.use_artifact(ARGS.class_model).download()
    # Downloading dataset and predictions
    full_dataset_path = run.use_artifact(ARGS.full_dataset).file()
    batch_prediction_path = run.use_artifact(ARGS.batch_prediction).file()
    
    LOGGER.info("Opening latest weather data")
    latest_data = pd.read_csv(full_dataset_path)

    LOGGER.info("Opening previous predictions")
    predicted_data = pd.read_csv(batch_prediction_path)

    LOGGER.info("Locating evaluation range")
    # Setting the times of predicted data that will be evaluated
    start = predicted_data.iloc[0]['time']
    end = predicted_data.iloc[7]['time']

    # Finding recorded data from the range that was predicted last week
    mask = (latest_data['time'] >= start) & (latest_data['time'] <= end)
    recorded_data = latest_data.loc[mask]

    LOGGER.info("Setting up real and prediction data")
    # Specifying predicted data
    predicted_data.rename(columns={'weathercode':'predicted_weathercode',
                                   'temperature_2m_max':'predicted_temperature_2m_max',
                                   'temperature_2m_min':'predicted_temperature_2m_min',
                                   'precipitation_sum':'predicted_precipitation_sum'}, inplace=True)

    # Merging real data with predictions
    performance = pd.merge(recorded_data, predicted_data, on='time', how='outer')

    LOGGER.info("Calculating prediction error")
    for objective, real, preds in zip(['weathercode_performace','max_temp_performace','min_temp_performace','precipitation_performace'],
                                      ['weathercode','temperature_2m_max','temperature_2m_min','precipitation_sum'],
                                      ['predicted_weathercode','predicted_temperature_2m_max','predicted_temperature_2m_min','predicted_precipitation_sum']):
        performance[objective] = abs(performance[real] - performance[preds])
    
    # Dropping irrelevant columns
    performance = performance.drop(['month-day_x','city_x','month-day_y','city_y'], axis=1)

    LOGGER.info("Average Weathercode error: %s", performance['weathercode_performace'].mean())
    LOGGER.info("Average Max temperature error: %s", performance['max_temp_performace'].mean())
    LOGGER.info("Average Min temperature error: %s", performance['min_temp_performace'].mean())
    LOGGER.info("Average Precipitation error: %s", performance['precipitation_performace'].mean())
    
    LOGGER.info("Setting up prediction for the next week")

    # Creating a date range for week past and week forward
    now = datetime.now()
    date_rng = pd.date_range(start=now - timedelta(days=7), end=now + timedelta(days=7) , freq='D', normalize=True)

    # Creating an input dataframe
    prediction = pd.DataFrame(date_rng, columns=['time'])
    prediction['time'] = pd.to_datetime(prediction['time'], format='%Y-%m-%d')
    prediction['month-day'] = prediction['time'].dt.strftime('%m.%d').astype(float)
    prediction.set_index('time', inplace=True)
    prediction['city'] = predicted_data['city'][0]

    LOGGER.info("Predicting temperature and precipitation")
    reg_model = mlflow.sklearn.load_model(reg_model_local_path)
    class_model = mlflow.sklearn.load_model(class_model_local_path)
    reg_pred = reg_model.predict(prediction)

    # Creating predicted columns
    prediction['temperature_2m_max'] = 0
    prediction['temperature_2m_min'] = 0
    prediction['precipitation_sum'] = 0

    # Adding predictions to the dataframe
    for i in range(len(reg_pred)):
        prediction['temperature_2m_max'][i] = reg_pred[i][0]
        prediction['temperature_2m_min'][i] = reg_pred[i][1]
        prediction['precipitation_sum'][i] = reg_pred[i][2]

    # Reordering columns
    prediction = prediction[['temperature_2m_max','temperature_2m_min','precipitation_sum','month-day','city']]

    LOGGER.info("Predicting the weathercode")
    class_pred = class_model.predict(prediction)

    # Creating predicted column
    prediction['weathercode'] = 0
    
    # Adding predictions to the dataframe
    for i in range(len(class_pred)):
        prediction['weathercode'][i] = class_pred[i]

    # Renaming predicted columns
    prediction = prediction.rename(columns = {'temperature_2m_max':'predicted_temperature_2m_max',
                                                'weathercode':'predicted_weathercode',
                                                'temperature_2m_min':'predicted_temperature_2m_min',
                                                'precipitation_sum':'predicted_precipitation_sum',})

    LOGGER.info("Average Weathercode prediction: %s", prediction['predicted_weathercode'].mean())
    LOGGER.info("Average Max temperature prediction: %s", prediction['predicted_temperature_2m_max'].mean())
    LOGGER.info("Average Min temperature prediction: %s", prediction['predicted_temperature_2m_min'].mean())
    LOGGER.info("Average Precipitation prediction: %s", prediction['predicted_precipitation_sum'].mean())

    LOGGER.info("Plotting next weeks weather prediction")
    # Selecting the last seven days of predicted data
    next_week = prediction.tail(7)

    # Calculating the average temperature range between predicted maximum and minimum temperatures
    temperature_range = (next_week['predicted_temperature_2m_max'] + next_week['predicted_temperature_2m_min']) / 2

    # Creating a new figure and axes for the plot
    fig, ax = plt.subplots()

    # Creating a secondary y-axis to show precipitation data
    ax2 = ax.twinx()

    # Filling the area between the predicted min and max temperatures with transparency
    ax.fill_between(next_week.index, next_week['predicted_temperature_2m_min'], next_week['predicted_temperature_2m_max'], alpha=0.5)

    # Plotting the mean temperature using markers and a line in blue color
    ax.plot(temperature_range, label='Mean temperature', marker='o', color='blue')

    # Plotting the predicted max temperature with red squares
    ax.plot(next_week['predicted_temperature_2m_max'], label='Max temperature', color='red', marker='s')

    # Plotting the predicted min temperature with blue triangles
    ax.plot(next_week['predicted_temperature_2m_min'], label='Min temperature', color='blue', marker='^')

    # Filling the area between the predicted precipitation and 0 with cyan color and transparency
    ax2.fill_between(next_week.index, next_week['predicted_precipitation_sum'], 0, alpha=0.5, color='cyan')

    # Plotting the predicted precipitation amount using markers and a line in cyan color
    ax2.plot(next_week['predicted_precipitation_sum'], label='Precipitation amount', color='cyan', marker=',')

    # Annotating the mean temperature values on the plot
    for x, y in zip(next_week.index, temperature_range):
        plt.annotate(str(round(y, 1)), xy=(x, y), ha='center', va='center')

    # Annotating the predicted precipitation values on the plot
    for x, y in zip(next_week.index, next_week['predicted_precipitation_sum']):
        plt.annotate(str(round(y, 1)), xy=(x, y), ha='center', va='center')

    # Setting labels for x and y axes, and title for the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Weather predictions for Bristol')

    # Setting y-axis limits for temperature and precipitation
    ax.set_ylim(0, 22)
    ax2.set_ylabel('Precipitation (mm)')
    ax2.set_ylim(0, 30)

    # Displaying gridlines and legends on the plot
    ax.grid(True)
    ax.legend()

    # Converting the plot to an image for logging purposes
    fig = wandb.Image(plt)

    # Logging the plot to the run
    run.log(({"plot": fig}))


    for file_name, k, desc in zip([performance, prediction],
                                  ['batch_prediction_performance.csv', 'batch_prediction.csv'],
                                  ['batch_prediction_performance','batch_prediction']):
        LOGGER.info("Uploading %s", desc)
        with tempfile.NamedTemporaryFile("w") as file:
            file_name.to_csv(file.name, index=True) # Saving as a temporary file
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

    LOGGER.info("Uploading log file of the run")
    log_file = f"../{datetime.now().strftime('%Y-%m-%d')}.log"
    log_artifact = wandb.Artifact(
        "log_file",
        type="log",
        description="Log file",
    )
    log_artifact.add_file(log_file)
    run.log_artifact(log_artifact)
    
    # Removing locally saved log file
    os.remove(f"../{datetime.now().strftime('%Y-%m-%d')}.log")

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step evaluates the predictions made previously by comparing them with the real data and makes new predictions for next week")

    PARSER.add_argument(
        "--reg_model",
        type=str,
        help="Input MLFlow Regression model",
        required=True
    )

    PARSER.add_argument(
        "--class_model",
        type=str,
        help="Input MLFlow Classification model",
        required=True
    )

    PARSER.add_argument(
        "--full_dataset",
        type=str,
        help="Input dataset with the latest data",
        required=True
    )

    PARSER.add_argument(
        "--batch_prediction",
        type=str,
        help="Input dataframe with previous predictions",
        required=True)

    ARGS = PARSER.parse_args()

    go(ARGS)