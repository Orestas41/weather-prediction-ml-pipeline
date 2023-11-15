"""
Merges all available data. Performs data cleaning and saves it in W&B
"""
# pylint: disable=E0401, W0621, C0103, R0914, R0915, E1101, C0200
from datetime import datetime
import logging
import wandb
import argparse
import pandas as pd
import os
import tempfile

# Set up logging
logging.basicConfig(
    filename=f"../{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()

def convert_date_column(data):
    """
    Convert a date column in a DataFrame to a datetime object.

    ARGS:
        data_frame: The DataFrame.
        format: The format of the date column.

    Returns:
        The DataFrame with the date column converted to a datetime object.
    """
    data['time'] = pd.to_datetime(
        data['time'], format='%Y-%m-%d')
    return data

def create_month_day_column(data):
    """
    Convert a date column in a DataFrame to a datetime object.

    ARGS:
        data_frame: The DataFrame.
        format: The format of the date column.

    Returns:
        The DataFrame with the date column converted to a datetime object.
    """
    data['month-day'] = data['time'].dt.strftime('%m.%d').astype(float)
    return data

def set_date_index(data):
    """
    Convert a date column in a DataFrame to a datetime object.

    ARGS:
        data_frame: The DataFrame.
        format: The format of the date column.

    Returns:
        The DataFrame with the date column converted to a datetime object.
    """
    data.set_index('time', inplace=True)
    return data

def merge_datasets(data, training_data):
    """
    Merges two DataFrames.

    ARGS:
        data_frame_1: The first DataFrame.
        data_frame_2: The second DataFrame.

    Returns:
        The merged DataFrame.
    """
    return pd.concat([training_data, data], axis=0)

def remove_na(data):
    """
    Removes rows with missing values from a DataFrame.

    ARGS:
        data_frame: The DataFrame.

    Returns:
        The DataFrame with missing values removed.
    """
    return data.dropna()

def drop_duplicates(data):
    """
    Drops duplicate rows from a DataFrame.

    ARGS:
        data_frame: The DataFrame.

    Returns:
        The DataFrame with duplicate rows dropped.
    """
    return data.drop_duplicates()

def sort_by_date(data):
    """
    Drops duplicate rows from a DataFrame.

    ARGS:
        data_frame: The DataFrame.

    Returns:
        The DataFrame with duplicate rows dropped.
    """
    data.index = pd.to_datetime(data.index)
    return data.sort_index()

def go(ARGS):
    """
    Combines all data processing functions and completes data pre-processing
    """

    # Creating instance
    run = wandb.init(
        job_type="pre-processing")
    run.config.update(ARGS)

    LOGGER.info("2 - Running pre-processing step")    

    LOGGER.info("Fetching %s artifact", ARGS.raw_data)
    raw_data_path = run.use_artifact(ARGS.raw_data).file()
    training_data_path = run.use_artifact(ARGS.training_data).file()

    LOGGER.info("Opening raw data file")
    data = pd.read_csv(raw_data_path)

    LOGGER.info("Converting time column into datetime format")
    data = convert_date_column(data)

    LOGGER.info("Creating month-day column")
    data = create_month_day_column(data)

    LOGGER.info("Merging new data with old training data")
    training_data = pd.read_csv(training_data_path)
    data = merge_datasets(data, training_data)

    LOGGER.info("Removing rows with missing values")
    data = remove_na(data)

    LOGGER.info("Setting date as index")
    data = set_date_index(data)

    LOGGER.info("Sorting by date")
    data = sort_by_date(data)

    LOGGER.info("Removing duplicate rows")
    data = drop_duplicates(data)

    LOGGER.info("Uploading training_data.csv file to W&B")
    with tempfile.NamedTemporaryFile("w") as file:
        data.to_csv(file.name, index=True)
        artifact = wandb.Artifact(
            ARGS.output_artifact,
            type=ARGS.output_type,
            description=ARGS.output_description,
        )
        artifact.add_file(file.name)
        run.log_artifact(artifact)

    LOGGER.info("Successfully pre-processed the data")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step merges and cleans the data")
    
    PARSER.add_argument("--raw_data", type=str, help="Input artifact to split")

    PARSER.add_argument("--training_data", type=str, help="Input artifact to split")

    PARSER.add_argument(
        "--output_artifact",
        type=str,
        help='Name of the output artifact',
        required=True
    )

    PARSER.add_argument(
        "--output_type",
        type=str,
        help='Type of the output artifact',
        required=True
    )

    PARSER.add_argument(
        "--output_description",
        type=str,
        help='Description of the output artifact',
        required=True
    )

    ARGS = PARSER.parse_args()

    go(ARGS)