"""
This script runs data tests
"""
# pylint: disable=E0401, E1101
import scipy.stats
import pandas as pd

def test_format(data):
    """
    Test the format of values is correct
    """
    # Convert the index of the DataFrame to a datetime
    data.index = pd.to_datetime(data.index)
    # Check if the index is in correct format
    assert isinstance(data.index, pd.DatetimeIndex)
    # Checking if columns that are not dates have either integer or float
    # values
    for column in data.columns:
        if column != 'time':
            assert data[column].dtype in (int, float)


def test_city_range(data):
    """
    Test the range of city values
    """
    assert data['city'].nunique() == 10
    # Checking that city values are between 1 and 10
    assert data['city'].between(1, 10).all()


def test_weathercode_range(data):
    """
    Test the range of weathercode values
    """
    # Checking that weathercode values are between 0 and 99
    assert data['weathercode'].between(0, 99).all()


def test_temperature_range(data):
    """
    Test the range of temperature values
    """
    # Checking that temperature values are between -30 and 45
    assert data['temperature_2m_max'].between(-30, 45).all()
    assert data['temperature_2m_min'].between(-30, 45).all()


def test_precipitation_range(data):
    """
    Test the range of precipitation values
    """
    # Checking that precipitation values are between 0 and 60
    assert 0 <= data['precipitation_sum'].any() <= 60


def test_similar_distrib(
        data: pd.DataFrame,
        ref_data: pd.DataFrame,
        threshold: float):
    """
    Applying a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    """
    columns_to_test = [
        'temperature_2m_min',
        'temperature_2m_max',
        'precipitation_sum']
    # Checking if the distribution difference is less than the k1 threshold
    for column in columns_to_test:
        dist1 = data[column].value_counts().sort_index()
        dist2 = ref_data[column].value_counts().sort_index()
        divergence = scipy.stats.entropy(dist1, dist2, base=2)
        assert divergence < threshold
