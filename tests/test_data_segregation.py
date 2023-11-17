"""
This test script tests the go function from data_segregation step
"""
# pylint: disable=E0401, C0413, W0621
import os
import sys
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Add the path to the directory containing the script you want to test
sys.path.append(
    "/home/orestas41/weather-prediction-ml-pipeline/data_segregation")

# Import the function to be tested
from run import go

@pytest.fixture
def mock_wandb():
    """Mock wandb connection"""
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init


@pytest.fixture
def mock_train_test_split():
    """Mock train test split"""
    with patch('run.train_test_split') as mock_split:
        yield mock_split


def test_go(mock_wandb, mock_train_test_split):
    """Test go function"""
    # Setting up test environment
    os.environ['TESTING'] = '1'

    # Setting up arguments
    args = MagicMock()
    args.input = "tests/mock_data.csv"
    args.test_size = 0.5

    # Mock wandb run
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    # Mock test data
    mock_artifact = MagicMock()
    mock_artifact.file.return_value = "tests/mock_data.csv"
    mock_run.use_artifact.return_value = mock_artifact

    # Mock data
    data = pd.read_csv('tests/mock_data.csv')
    data.set_index('time', inplace=True)

    # Mock train test split
    mock_dataframe = data
    mock_train_test_split.return_value = (mock_dataframe, mock_dataframe)

    # Run the function
    go(args)
