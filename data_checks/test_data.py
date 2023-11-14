"""
This script runs data tests
"""
# pylint: disable=E0401, E1101
import logging
from datetime import datetime
import wandb
import scipy
import pandas as pd

# Setting up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.ERROR)
LOGGER = logging.getLogger()

RUN = wandb.init(
    job_type="data_check")

LOGGER.info("3 - Running data checks")

def test_format(data):
    """
    Test the format of values is correct
    """
    LOGGER.info("Testing if the format of the values are correct")
    # Convert the index of the DataFrame to a datetime
    # Check if the index is in correct format
    data.index = pd.to_datetime(data.index)
    assert isinstance(data.index, pd.DatetimeIndex)
    # Checking if columns that are not dates have either integer or float
    # values
    for column in data.columns:
        if column != 'time':
            assert data[column].dtype in (int, float)

def test_city_range(data):
    """
    Test the range of winner values
    """
    LOGGER.info("Testing if the values of Winner column are correct")
    assert data['city'].nunique() == 10
    # Checking that city values are between 1 and 10
    assert data['city'].min() >= 1 and data['city'].max() <= 10

def test_weathercode_range(data):
    """
    Test the range of winner values
    """
    LOGGER.info("Testing if the values of Winner column are correct")
    # Checking that winner values are between 0 and 1
    assert data['weathercode'].min() >= 0 and data['weathercode'].max() <= 99

def test_temperature_range(data):
    """
    Test the range of winner values
    """
    LOGGER.info("Testing if the values of Winner column are correct")
    # Checking that city values are between 1 and 10
    assert -30 < data['temperature_2m_max'].any() < 45
    assert -30 < data['temperature_2m_min'].any() < 45

def test_precipitation_range(data):
    """
    Test the range of winner values
    """
    LOGGER.info("Testing if the values of Winner column are correct")
    # Checking that city values are between 1 and 10
    assert 0 <= data['precipitation_sum'].any() < 60
    assert 0 <= data['precipitation_sum'].any() < 60


def test_similar_distrib(
    data: pd.DataFrame,
    ref_data: pd.DataFrame,
    threshold: float):
    """
    #Applying a threshold on the KL divergence to detect if the distribution of the new data is
    #significantly different than that of the reference dataset
    """
    LOGGER.info(
        "Testing of the distribution of the dataset is similar to what is expected")
    dist1 = data['precipitation_sum'].value_counts().sort_index()
    dist2 = ref_data['precipitation_sum'].value_counts().sort_index()
    # Checking if the distribution difference is less than the k1 threshold
    assert scipy.stats.entropy(dist1, dist2, base=2) < threshold

    LOGGER.info("Finished data checks")