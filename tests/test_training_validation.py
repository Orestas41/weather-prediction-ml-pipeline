import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import json
import tempfile
from datetime import datetime
import sys

# Add the path to the directory containing the script you want to test
sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/training_validation")

# Import the function to be tested
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
    
    # Replace 'YourArgumentsClass' with the actual class used in your script for arguments
    temp_file = tempfile.NamedTemporaryFile(mode='w+')
    json.dump({'param':'value'}, temp_file)
    temp_file.flush()
    model_config = os.path.abspath(temp_file.name)

    args = MagicMock()
    args.trainval_artifact = "data/trainval.csv"
    args.val_size = 0.5
    args.model_config = model_config
    args.output_artifact = "output_model"

    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    mock_artifact = MagicMock()
    mock_artifact.file.return_value = "data/trainval.csv"
    mock_run.use_artifact.return_value = mock_artifact

    data = {"time": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
            "weathercode": [1, 2, 3, 4],
            "temperature_2m_max": [2, 3, 4, 5],
            "temperature_2m_min": [1, 2, 3, 4],
            "precipitation_sum": [1, 2, 3, 4],
            "city": [1, 2, 3, 4],
            "month-day": [1001, 1002, 1003, 1004],
            }
    
    data['time'] = pd.to_datetime(
        data['time'], format='%Y-%m-%d')
    
    mock_dataframe = pd.DataFrame(data)
    x_mock_dataframe = mock_dataframe.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
    y_mock_dataframe = mock_dataframe[['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]
    mock_train_test_split.return_value = (x_mock_dataframe.head(2), x_mock_dataframe.tail(2), y_mock_dataframe.head(2), y_mock_dataframe.tail(2))  # Replace with your desired dataframe

    # Run the function
    go(args)

    # Assertions
    assert mock_run.use_artifact.called_with("data/trainval.csv")
    assert mock_train_test_split.called_with(mock_dataframe, test_size=0.5)
    assert mock_run.log_artifact.called_with('output_model')

    assert os.path.exists('model_dir')