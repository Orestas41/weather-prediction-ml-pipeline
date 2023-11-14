import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import json
import tempfile
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
    json.dump({'n_estimators': 10}, temp_file)
    temp_file.flush()
    reg_model_config = os.path.abspath(temp_file.name)
    class_model_config = os.path.abspath(temp_file.name)

    args = MagicMock()
    args.trainval_artifact = "tests/mock_data.csv"
    args.val_size = 0.5
    args.reg_model_config = reg_model_config
    args.class_model_config = class_model_config
    args.output_artifact = "output_model"

    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    mock_artifact = MagicMock()
    mock_artifact.file.return_value = "tests/mock_data.csv"
    mock_run.use_artifact.return_value = mock_artifact

    data = pd.read_csv('tests/mock_data.csv')

    mock_dataframe = pd.DataFrame(data)
    mock_train_test_split.return_value = (mock_dataframe.head(2), mock_dataframe.tail(2), mock_dataframe.head(2), mock_dataframe.tail(2))

    # Run the function
    go(args)

    # Assertions
    assert mock_run.use_artifact.called_with("tests/mock_data.csv")
    assert mock_train_test_split.called_with(mock_dataframe, test_size=0.5)
    assert mock_run.log_artifact.called_with('output_model')

    #assert os.path.exists('training_validation/reg_model_dir')
    #assert os.path.exists('training_validation/class_model_dir')