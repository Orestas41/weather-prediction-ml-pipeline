"""
Merges all available data. Performs data cleaning and saves it in W&B
"""
# pylint: disable=E0401, W0621, C0103, R0914, R0915, E1101, C0200
"""import os
import pickle
from datetime import datetime
import argparse
import logging
import yaml
import numpy as np
from sklearn.preprocessing import LabelEncoder
import wandb"""
import pandas as pd
import json

# Set up logging
"""logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()"""

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
    data['month-day'] = data['time'].dt.strftime('%m-%d')
    data['month-day'] = pd.to_datetime(data['month-day'], format='%m-%d')
    data['month-day'] = pd.to_datetime(data['month-day']).dt.strftime('%m%d').astype(int)
    return data

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


#def go(ARGS):
"""
Combines all data processing functions and completes data pre-processing
"""
"""
# Load config.yaml and get input and output paths
with open('../config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# Setting-up directory paths
input_folder_path = config['directories']['input_folder_path']
output_folder_path = config['directories']['output_folder_path']

# Setting-up ingested file recording
file_record = open(
    f"../reports/ingestedfiles/{datetime.now().strftime('%Y-%m-%d')}.txt",
    "w")"""

# Creating instance
"""run = wandb.init(
    job_type="pre-processing")
run.config.update(ARGS)

LOGGER.info("2 - Running pre-processing step")

LOGGER.info("Merging multiple dataframes")

# Get the datasets
raw_data = pd.DataFrame()
raw_datasets = []
for dataset in os.listdir('../' + input_folder_path):
    raw_datasets.append(pd.read_csv(os.path.join(
        '../' + input_folder_path, dataset), header=None))
    file_record.write(str(dataset) + '\n')"""

with open('./data/new_data.json') as f:
    data = json.load(f)

data = pd.DataFrame(data['daily'])
training_data = pd.read_csv('./data/training_data.csv')

#LOGGER.info("Converting Date column into datetime format")
data = convert_date_column(data)

#LOGGER.info("Converting Date column into datetime format")
data = create_month_day_column(data)

# Merge the datasets
data = merge_datasets(data, training_data)

# Remove duplicate rows
data = drop_duplicates(data)

#LOGGER.info("Removeing rows with missing values")
data = remove_na(data)

# Set date as index
data = set_date_index(data)

# Save merged datasets as one file
data.to_csv(f'./data/training_data.csv')

"""LOGGER.info("Uploading processed_data.csv file to W&B")
artifact = wandb.Artifact(
    ARGS.output_artifact,
    type=ARGS.output_type,
    description=ARGS.output_description,
)
artifact.add_file(f'../{output_folder_path}/processed_data.csv')
run.log_artifact(artifact)

LOGGER.info("Successfully pre-processed the data")"""

"""
if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step merges and cleans the data")

    PARSER.add_argument(
        "--input_artifact",
        type=str,
        help='Fully-qualified name for the input artifact',
        required=True
    )

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

    go(ARGS)"""