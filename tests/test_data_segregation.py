import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import os
import sys

# Add the path to the directory containing the script you want to test
sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/data_segregation")

from run import go

@pytest.fixture
def mock_wandb():
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init

@pytest.fixture
def mock_train_test_split():
    with patch('run.train_test_split') as mock_split:
        yield mock_split

def test_go(mock_wandb, mock_train_test_split):
    # Set up test data
    os.environ['TESTING'] = '1'
    args = MagicMock()
    args.input = "data/training_data.csv"
    args.test_size = 0.5

    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    mock_artifact = MagicMock()
    mock_artifact.file.return_value = "data/training_data.csv"
    mock_run.use_artifact.return_value = mock_artifact

    mock_dataframe = pd.DataFrame({"time": ["2022-01-01", "2022-01-02"], "value": [1, 2]})
    mock_train_test_split.return_value = (mock_dataframe, mock_dataframe)  # Replace with your desired dataframe

    # Run the function
    go(args)

    # Assertions
    assert mock_run.use_artifact.called_with("data/training_data.csv")
    assert mock_train_test_split.called_with(mock_dataframe, test_size=0.5)
    assert mock_run.log_artifact.called_with('trainval_data.csv')

    assert os.path.exists('data/trainval.csv')
    assert os.path.exists('data/test.csv')