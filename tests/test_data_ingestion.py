"""
This test script tests the go function from data_ingestion step
"""
# pylint: disable=E0401, W0621, C0413
import os
import sys
import json
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import pytest

# Add the path to the directory containing the script you want to test
main_path = os.getcwd()
sys.path.append(
    f"{main_path}/data_ingestion")

# Import the function to be tested
from run import go

@pytest.fixture
def mock_wandb():
    """Mock wandb connection"""
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init


@pytest.fixture
def mock_http_client():
    """Mock HTTPS connection"""
    with patch('run.http.client.HTTPSConnection') as mock_conn:
        yield mock_conn


@pytest.fixture
def mock_date():
    """Mock date"""
    with patch('run.datetime') as mock_datetime:
        yield mock_datetime


@pytest.fixture
def mock_read():
    """Mock date"""
    with patch('run.pd.read_csv') as mock_read:
        yield mock_read


def test_go(mock_http_client, mock_wandb, mock_date, mock_read):
    """Test go function"""
    # Setting up test environment
    os.environ['TESTING'] = '1'

    # Setting up arguments
    args = MagicMock()
    args.step_description = 'Testing data_ingestion'
    args.hostname = "example.com"

    # Mock wandb run
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    # Mock API request
    mock_res = MagicMock()
    example_response = {
        "daily": {
            "time": ["2023-01-01"],
            "weathercode": [1],
            "temperature_2m_max": [1],
            "temperature_2m_min": [1],
            "precipitation_sum": [1]}}
    mock_res.read.return_value = json.dumps(example_response).encode("utf-8")
    mock_conn_instance = mock_http_client.return_value
    mock_conn_instance.getresponse.return_value = mock_res

    # Mock ingestion records
    mock_records = pd.DataFrame(
        {'Date': ['2023-01-01'], 'Start': ['2023-01-01'], 'End': ['2023-01-01']})
    mock_read.return_value = mock_records

    # Mock the current date
    date = datetime(2023, 1, 1)
    mock_date.now.return_value = date

    # Run the function
    go(args)
