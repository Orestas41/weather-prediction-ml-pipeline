import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from datetime import datetime
import sys

# Add the path to the directory containing the script you want to test
sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/batch_prediction")

# Import the function to be tested
from run import go

@pytest.fixture
def mock_wandb():
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init

@pytest.fixture
def mock_mlflow():
    with patch('run.mlflow.sklearn.load_model') as mock_load_model:
        yield mock_load_model

def test_go(mock_wandb, mock_mlflow):
    # Set up test data
    os.environ['TESTING'] = '1'

    args = MagicMock()
    args.mlflow_model = "model_export"
    args.full_dataset = 'tests/mock_data.csv'

    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    mock_artifact = MagicMock()
    mock_artifact.file.return_value = 'tests/mock_data.csv'
    mock_run.use_artifact.return_value = mock_artifact

    # Mock the mlflow.sklearn.load_model function
    mock_model = MagicMock()
    mock_mlflow.return_value = mock_model

    preds = [[1,1.1,0.1,1.1],
            [2,2.2,0.2,2.2],
            [3,3.3,0.3,3.3],
            [4,4.4,0.4,4.4],
            [5,5.5,0.5,5.5],
            [6,6.6,0.6,6.6],
            [7,7.7,0.7,7.7],
            [8,8.8,0.8,8.8],]

    mock_model.predict.return_value = preds

    # Run the function
    go(args)

    # Assertions
    assert mock_run.use_artifact.called_with("your_full_dataset")
    assert mock_mlflow.called_with("your_mlflow_model")
    assert mock_model.predict.called  # Add more assertions based on your specific implementation
