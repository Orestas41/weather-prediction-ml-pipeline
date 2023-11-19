"""
This test script tests the go function from model_test step
"""
# pylint: disable=E0401, C0413, W0621
import os
import sys
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Add the path to the directory containing the script you want to test
main_path = os.getcwd()
sys.path.append(
    f"{main_path}/model_test")

# Import the function to be tested
from run import go

@pytest.fixture
def mock_wandb():
    """Mock wandb connection"""
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init


@pytest.fixture
def mock_model():
    """Mock model"""
    with patch('run.mlflow.sklearn.load_model') as mock_load_model:
        yield mock_load_model


@pytest.fixture
def mock_mae():
    """Mock mean absolute error calculation"""
    with patch('run.mean_absolute_error') as mock_mae:
        yield mock_mae


def test_go(mock_wandb, mock_model, mock_mae):
    """Test go function"""
    # Setting up test environment
    os.environ['TESTING'] = '1'

    # Setting up arguments
    args = MagicMock()
    args.reg_model = "reg_model"
    args.class_model = 'class_model'
    args.test_dataset = "tests/mock_data.csv"
    args.performance_records = 'model_performance.csv:latest'

    # Mock wandb run
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    # Mock test data
    mock_reg_artifact = MagicMock()
    mock_class_artifact = MagicMock()
    mock_reg_artifact.file.return_value = "tests/mock_data.csv"
    mock_class_artifact.file.return_value = "tests/mock_perf.csv"
    mock_run.use_artifact.side_effect = [
        mock_reg_artifact, mock_class_artifact]

    # Mock models
    mock_reg_model = MagicMock()
    mock_class_model = MagicMock()
    mock_model.side_effect = [mock_reg_model, mock_class_model]

    # Mock data
    data = pd.read_csv("tests/mock_data.csv")
    data.set_index('time', inplace=True)

    # Set return values for predictions
    mock_reg_model.predict.return_value = data[[
        'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]
    mock_class_model.predict.return_value = data['weathercode']

    # Set return values for model score
    mock_reg_model.score.return_value = 1.1
    mock_class_model.score.return_value = 1.1

    # Set return values for mean absolute error
    mock_mae.return_value = 1.1

    # Run the function
    go(args)
