"""
Merges all available data. Performs data cleaning and saves it in W&B
"""
# pylint: disable=E0401, W0621, C0103, R0914, R0915, E1101, C0200
import os
import pickle
from datetime import datetime
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import wandb

# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def merge_datasets(data_frame_1, data_frame_2):
    """
    Merges two DataFrames.

    ARGS:
        data_frame_1: The first DataFrame.
        data_frame_2: The second DataFrame.

    Returns:
        The merged DataFrame.
    """
    return pd.concat([data_frame_1, data_frame_2], axis=0)


def drop_duplicates(data_frame):
    """
    Drops duplicate rows from a DataFrame.

    ARGS:
        data_frame: The DataFrame.

    Returns:
        The DataFrame with duplicate rows dropped.
    """
    return data_frame.drop_duplicates()


def remove_na(data_frame):
    """
    Removes rows with missing values from a DataFrame.

    ARGS:
        data_frame: The DataFrame.

    Returns:
        The DataFrame with missing values removed.
    """
    return data_frame.dropna()


def convert_date_column(data_frame):
    """
    Convert a date column in a DataFrame to a datetime object.

    ARGS:
        data_frame: The DataFrame.
        format: The format of the date column.

    Returns:
        The DataFrame with the date column converted to a datetime object.
    """
    data_frame['Date'] = pd.to_datetime(
        data_frame['Date'], format='%Y-%m-%d, %H:%M')
    return data_frame


def sort_dataframe(data_frame, col):
    """
    Sorts a DataFrame by a column.

    ARGS:
        data_frame: The DataFrame.
        col: The column to sort by.

    Returns:
        The sorted DataFrame.
    """
    return data_frame.sort_values(by=col)


def encode_team_names(data_frame, encoder):
    """
    Encodes the team names in a DataFrame using a LabelEncoder.

    ARGS:
        data_frame: The DataFrame.
        encoder: A LabelEncoder object.

    Returns:
        The DataFrame with the team names encoded.
    """
    data_frame['Home'] = encoder.transform(data_frame['Home'])
    data_frame['Away'] = encoder.transform(data_frame['Away'])
    return data_frame


def go(ARGS):
    """
    Combines all data processing functions and completes data pre-processing
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
        "w")

    # Creating instance
    run = wandb.init(
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
        file_record.write(str(dataset) + '\n')

    # Merge the datasets
    for i in range(len(raw_datasets)):
        raw_data = merge_datasets(raw_data, raw_datasets[i])

    # Remove duplicate rows
    raw_data = drop_duplicates(raw_data)

    # Save merged datasets as one file
    raw_data.to_csv(f'../{output_folder_path}/raw_data.csv', index=None)

    LOGGER.info("Adding headers")
    raw_data.columns = ["Date", "Blank", "Home", "Result", "Away", "Location"]

    LOGGER.info("Removeing rows with missing values")
    raw_data = remove_na(raw_data)

    LOGGER.info("Converting Date column into datetime format")
    raw_data = convert_date_column(raw_data)

    LOGGER.info("Sorting dataframe by date")
    raw_data = sort_dataframe(raw_data, 'Date')

    # Converting dates to  timestamps
    raw_data['Date'] = raw_data['Date'].astype(int) / 10**18

    # Convert Result column into separate columns for Home and Away goals
    raw_data[['Home Result', 'Away Result']] = raw_data['Result'].str.split(
        ' : ', expand=True).astype(int)

    # Create Winner column with the team that won or draw
    raw_data['Winner'] = np.where(
        raw_data['Home Result'] > raw_data['Away Result'], 0, np.where(
            raw_data['Home Result'] < raw_data['Away Result'], 1, 0.5))

    LOGGER.info("Encoding unique strings")
    encoder = LabelEncoder()
    encoder.fit(raw_data['Home'])
    raw_data = encode_team_names(raw_data, encoder)

    LOGGER.info('Saving encoder locally')
    encoder_file = 'encoder.pkl'
    with open(encoder_file, 'wb') as file:
        pickle.dump(encoder, file)

    LOGGER.info('Saving encoder to wandb')
    encoder_artifact = wandb.Artifact(
        'encoder',
        type='encoder'
    )
    encoder_artifact.add_file('encoder.pkl')
    run.log_artifact(encoder_artifact)

    LOGGER.info("Dropping unnecessary columns")
    raw_data = raw_data.drop(['Blank', 'Location', 'Result',
                              'Home Result', 'Away Result'], axis=1)

    LOGGER.info("Saving dataframe as a csv file")
    raw_data.to_csv(f'../{output_folder_path}/processed_data.csv', index=None)

    LOGGER.info("Uploading processed_data.csv file to W&B")
    artifact = wandb.Artifact(
        ARGS.output_artifact,
        type=ARGS.output_type,
        description=ARGS.output_description,
    )
    artifact.add_file(f'../{output_folder_path}/processed_data.csv')
    run.log_artifact(artifact)

    LOGGER.info("Successfully pre-processed the data")


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

    go(ARGS)