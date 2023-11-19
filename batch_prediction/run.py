"""
Evaluating previous weather prediction results against real world data.
Batch predicting the weather of last and next week
"""
# pylint: disable=E0401, W0621, R0914, E1101, C0200, C0103, W0106, R0915
import os
import logging
import argparse
from datetime import datetime, timedelta
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import wandb

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
        "Downloading %s and %s",
        ARGS.reg_model,
        ARGS.class_model,
    )

    # Downloading model artifacts
    reg_model_local_path = run.use_artifact(ARGS.reg_model).download()
    class_model_local_path = run.use_artifact(ARGS.class_model).download()

    LOGGER.info("Setting up prediction for the next week")

    # Creating a date range for week past and week forward

    now = datetime.now()
    new_date_range = pd.date_range(
        start=now,
        end=now + timedelta(days=7),
        freq='D', normalize=True)

    # Creating an input dataframe
    prediction = pd.DataFrame(new_date_range, columns=['time'])
    prediction['time'] = pd.to_datetime(prediction['time'], format='%Y-%m-%d')
    prediction['month-day'] = prediction['time'].dt.strftime(
        '%m.%d').astype(float)
    prediction.set_index('time', inplace=True)
    prediction['city'] = 7

    LOGGER.info("Predicting temperature and precipitation")
    reg_model = mlflow.sklearn.load_model(reg_model_local_path)
    class_model = mlflow.sklearn.load_model(class_model_local_path)
    reg_pred = reg_model.predict(prediction)

    # Creating predicted columns
    prediction[['temperature_2m_max',
                'temperature_2m_min',
                'precipitation_sum']] = reg_pred

    # Reordering columns
    prediction = prediction[['temperature_2m_max',
                             'temperature_2m_min',
                             'precipitation_sum',
                             'month-day',
                             'city']]

    LOGGER.info("Predicting the weathercode")
    class_pred = class_model.predict(prediction)

    # Creating predicted column
    prediction['weathercode'] = class_pred

    LOGGER.info("\n%s", prediction.to_string(index=True))

    LOGGER.info("Plotting next weeks weather prediction")
    # Selecting the last seven days of predicted data
    next_week = prediction.tail(7)

    # Calculating the average temperature range between predicted maximum and
    # minimum temperatures
    temperature_range = (next_week['temperature_2m_max'] +
                         next_week['temperature_2m_min']) / 2

    # Creating a new figure and axes for the plot
    fig, ax = plt.subplots()

    # Creating a secondary y-axis to show precipitation data
    ax2 = ax.twinx()

    # Filling the area between the predicted min and max temperatures with
    # transparency
    ax.fill_between(
        next_week.index,
        next_week['temperature_2m_min'],
        next_week['temperature_2m_max'],
        alpha=0.5)

    # Plotting the mean temperature using markers and a line in blue color
    ax.plot(
        temperature_range,
        label='Mean temperature',
        marker='o',
        color='blue')

    # Plotting the predicted max temperature with red squares
    ax.plot(next_week['temperature_2m_max'],
            label='Max temperature', color='red', marker='s')

    # Plotting the predicted min temperature with blue triangles
    ax.plot(
        next_week['temperature_2m_min'],
        label='Min temperature',
        color='blue',
        marker='^')

    # Filling the area between the predicted precipitation and 0 with cyan
    # color and transparency
    ax2.fill_between(
        next_week.index,
        next_week['precipitation_sum'],
        0,
        alpha=0.5,
        color='cyan')

    # Plotting the predicted precipitation amount using markers and a line in
    # cyan color
    ax2.plot(
        next_week['precipitation_sum'],
        label='Precipitation amount',
        color='cyan',
        marker=',')

    # Annotating the predicted precipitation values on the plot
    for x, y in zip(next_week.index, next_week['precipitation_sum']):
        plt.annotate(str(round(y, 1)), xy=(x, y), ha='center', va='center')

    # Setting labels for x and y axes, and title for the plot
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Weather predictions for Bristol')

    # Setting y-axis limits for temperature and precipitation
    ax.set_ylim((next_week['temperature_2m_min'].min() - 5), 
                (next_week['temperature_2m_max'].max() + 3))
    ax2.set_ylabel('Precipitation (mm)')
    ax2.set_ylim(0, 30)

    # Displaying gridlines and legends on the plot
    ax.grid(True)
    ax.legend()

    # Converting the plot to an image for logging purposes
    fig = wandb.Image(plt)

    # Logging the plot to the run
    run.log(({"plot": fig}))

    LOGGER.info("Batch tour evaluations and predictions finished")

    LOGGER.info("Uploading log file of the run and deleting locally stored file")
    log_file = f"../{now.strftime('%Y-%m-%d')}.log"
    log_artifact = wandb.Artifact(
        "log_file",
        type="log",
        description="Log file",
    )
    if not os.getenv('TESTING'):
        log_artifact.add_file(log_file)
        run.log_artifact(log_artifact)
        # Removing locally saved log file
        os.remove(f"../{now.strftime('%Y-%m-%d')}.log")
    else:
        pass


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="Evaluate predictions made previously with the real data, make new predictions")

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

    ARGS = PARSER.parse_args()

    go(ARGS)
