"""
This test script tests the go function from training_validation step
"""
# pylint: disable=E0401, C0413, W0621, W1514
import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Add the path to the directory containing the script you want to test
main_path = os.getcwd()
sys.path.append(
    f"{main_path}/training_validation")

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

@pytest.fixture
def mock_save_model():
    """Mock saving model"""
    with patch('run.mlflow.sklearn.save_model') as mock_save_model:
        yield mock_save_model


def test_go(mock_wandb, mock_train_test_split, mock_save_model):
    """Test go function"""
    # Setting up test environment
    os.environ['TESTING'] = '1'

    # Setting up mock configuration file
    tfile = tempfile.mkstemp(suffix='.json')
    file_dir, temp_path = tfile
    with open(file_dir, 'w') as file:
        json.dump({'n_estimators': 10}, file)
    reg_model_config = temp_path
    class_model_config = temp_path

    # Setting up arguments
    args = MagicMock()
    args.trainval_artifact = "tests/mock_data.csv"
    args.val_size = 0.5
    args.reg_config = reg_model_config
    args.class_config = class_model_config
    args.output_artifact = "output_model"

    # Mock wandb run
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    # Mock test data
    mock_artifact = MagicMock()
    mock_artifact.file.return_value = "tests/mock_data.csv"
    mock_run.use_artifact.return_value = mock_artifact

    mock_model = MagicMock()
    mock_save_model.return_value = mock_model

    # Mock data
    data = pd.read_csv('tests/mock_data.csv')
    data.set_index('time', inplace=True)

    # Mock train-test split
    mock_dataframe = pd.DataFrame(data)
    mock_train_test_split.return_value = (
        mock_dataframe.head(2),
        mock_dataframe.tail(2),
        mock_dataframe.head(2),
        mock_dataframe.tail(2))

    # Run the function
    go(args)
