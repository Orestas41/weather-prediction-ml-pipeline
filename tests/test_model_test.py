import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import json
import tempfile
from datetime import datetime
import sys

# Add the path to the directory containing the script you want to test
sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/model_test")

# Import the function to be tested
from run import go

@pytest.fixture
def mock_wandb():
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init

@pytest.fixture
def mock_model():
    with patch('run.mlflow.sklearn.load_model') as mock_load_model:
        yield mock_load_model

def test_go(mock_wandb, mock_model):
    # Set up test data
    os.environ['TESTING'] = '1'

    args = MagicMock()
    args.reg_model = "reg_model"
    args.class_model = 'class_model'
    args.test_dataset = "tests/mock_data.csv"

    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    mock_artifact = MagicMock()
    mock_artifact.file.return_value = "tests/mock_data.csv"
    mock_run.use_artifact.return_value = mock_artifact

    # Mock the mlflow.sklearn.load_model function
    model = MagicMock()
    mock_model.return_value = model
    mock_model = MagicMock()
    mock_model.return_value = model

    data = pd.read_csv('tests/mock_data.csv')

    model.predict.return_value = data[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]
    model.predict.return_value = data['weathercode']
    
    model.score.return_value = 1.1

    #
    model.score.return_value = 1.1

    # Run the function
    go(args)

    # Assertions
    assert mock_run.use_artifact.called_with("your_test_dataset")
    assert mock_mlflow.called_with("your_mlflow_model")
    assert mock_model.predict.called  # Add more assertions based on your specific implementation
