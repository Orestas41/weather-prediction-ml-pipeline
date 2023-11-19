"""
This test script tests the go function from batch_prediction step
"""
# pylint: disable=E0601, E0401, W0621, C0413
import os
import sys
from unittest.mock import patch, MagicMock
import pytest

# Add the path to the directory containing the script you want to test
main_path = os.getcwd()
sys.path.append(
    f"{main_path}/batch_prediction")

# Import the function to be tested
from run import go

@pytest.fixture
def mock_wandb():
    """Mock wandb connection"""
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init


@pytest.fixture
def mock_mlflow():
    """Mock model"""
    with patch('run.mlflow.sklearn.load_model') as mock_load_model:
        yield mock_load_model


def test_go(mock_wandb, mock_mlflow):
    """Test go function"""
    # Setting up test environment
    os.environ['TESTING'] = '1'

    # Setting up arguments
    args = MagicMock()
    args.reg_model = "reg_model:prod"
    args.class_model = "class_model:prod"

    # Mock wandb run
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    # Mock test data
    mock_artifact = MagicMock()
    mock_artifact.file.return_value = 'tests/mock_data.csv'
    mock_run.use_artifact.return_value = mock_artifact

    # Mock the mlflow.sklearn.load_model function
    mock_model = MagicMock()
    mock_mlflow.return_value = mock_model

    # Mock prediction
    reg_preds = [[1.1, 0.1, 1.1],
                 [2.2, 0.2, 2.2],
                 [3.3, 0.3, 3.3],
                 [4.4, 0.4, 4.4],
                 [5.5, 0.5, 5.5],
                 [6.6, 0.6, 6.6],
                 [7.7, 0.7, 7.7],
                 [8.8, 0.8, 8.8]]

    # Using mock predictions in the script
    mock_model.predict.return_value = reg_preds

    # Run the function
    go(args)
