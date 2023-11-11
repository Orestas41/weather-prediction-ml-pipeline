import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from tempfile import NamedTemporaryFile
import sys

# Add the path to the directory containing the script you want to test
sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/data_checks")

# Import the module you want to test
from conftest import pytest_addoption
from test_data import (
    test_format,
    test_city_range,
    test_weathercode_range,
    test_temperature_range,
    test_precipitation_range,
    #test_similar_distrib,
)

@pytest.fixture
def mock_wandb():
    with patch('conftest.wandb.init') as mock_wandb_init:
        yield mock_wandb_init

@pytest.fixture
def mock_run(mock_wandb):
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run
    return mock_run

def test_pytest_addoption():
    # Mocking the pytest parser
    parser = MagicMock()
    pytest_addoption(parser)

    # Assert that the parser has the expected calls
    assert call('--csv', action='store') in parser.addoption.call_args_list
    assert call('--ref', action='store') in parser.addoption.call_args_list
    assert call('--kl_threshold', action='store', type=float) in parser.addoption.call_args_list

# Mock data for testing
@pytest.fixture
def data(mock_run):
    return pd.read_csv('tests/mock_data.csv')

@pytest.fixture
def mock_ref_data(mock_run):
    data = pd.read_csv('tests/mock_data.csv')
    return data

@pytest.fixture
def mock_threshold():
    return 0.5

def test_test_format(data):
    test_format(data)  # No assertion errors indicate success

def test_test_city_range(data):
    test_city_range(data)  # No assertion errors indicate success

def test_test_weathercode_range(data):
    test_weathercode_range(data)  # No assertion errors indicate success

def test_test_temperature_range(data):
    test_temperature_range(data)  # No assertion errors indicate success

def test_test_precipitation_range(data):
    test_precipitation_range(data)  # No assertion errors indicate success



"""def test_test_similar_distrib(mock_data, mock_ref_data, mock_threshold):
    with patch('your_module.LOGGER') as mock_logger:
        test_similar_distrib(mock_data, mock_ref_data, mock_threshold)
        
        # Add assertions based on your specific implementation
    assert mock_logger.info.called_with("Testing of the distribution of the dataset is similar to what is expected")
    assert mock_logger.info.called_with("Finished data checks")
"""

