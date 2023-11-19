"""
This test script tests the pytest functions from data_checks step
"""
# pylint: disable=E0601, E0401, W0621, C0413, W0613
import os
import sys
from unittest.mock import patch, MagicMock, call
import pandas as pd
import pytest

# Add the path to the directory containing the script you want to test
main_path = os.getcwd()
sys.path.append(
    f"{main_path}/data_checks")

# Import the module you want to test
from conftest import pytest_addoption
from test_data import (
    test_format,
    test_city_range,
    test_weathercode_range,
    test_temperature_range,
    test_precipitation_range,
    test_similar_distrib,
)

@pytest.fixture
def mock_wandb():
    """Mock wandb connection"""
    with patch('conftest.wandb.init') as mock_wandb_init:
        yield mock_wandb_init


@pytest.fixture
def mock_run(mock_wandb):
    """Mock wandb run"""
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run
    return mock_run


def test_pytest_addoption():
    """Mocking the pytest parser"""
    parser = MagicMock()
    pytest_addoption(parser)

    # Assert that the parser has the expected calls
    assert call('--csv', action='store') in parser.addoption.call_args_list
    assert call('--ref', action='store') in parser.addoption.call_args_list
    assert call(
        '--kl_threshold',
        action='store',
        type=float) in parser.addoption.call_args_list

# Mock data for testing


@pytest.fixture
def data(mock_run):
    """Initiating mock data as data"""
    return pd.read_csv('tests/mock_data.csv')


@pytest.fixture
def ref_data(mock_run):
    """Initiating mock data as reference data"""
    data = pd.read_csv('tests/mock_data.csv')
    return data


@pytest.fixture
def threshold(mock_run):
    """Initiating mock threshold"""
    return 0.5


def test_test_format(data):
    """Test test_format function"""
    test_format(data)  # No assertion errors indicate success


def test_test_city_range(data):
    """Test test_city_range function"""
    test_city_range(data)  # No assertion errors indicate success


def test_test_weathercode_range(data):
    """Test test_weathercode_range function"""
    test_weathercode_range(data)  # No assertion errors indicate success


def test_test_temperature_range(data):
    """Test test_temperature_range function"""
    test_temperature_range(data)  # No assertion errors indicate success


def test_test_precipitation_range(data):
    """Test test_precipitation_range function"""
    test_precipitation_range(data)  # No assertion errors indicate success


def test_test_similar_distrib(data, ref_data, threshold):
    """Test test_similar_distrib function"""
    test_similar_distrib(
        data,
        ref_data,
        threshold)  # No assertion errors indicate success
