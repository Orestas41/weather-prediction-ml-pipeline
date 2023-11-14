import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
import os
import sys

sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/data_ingestion")

from run import go

@pytest.fixture
def mock_http_client():
    with patch('run.http.client.HTTPSConnection') as mock_conn:
        yield mock_conn

def test_go(mock_http_client):
    # Set up test data
    os.environ['TESTING'] = '1'
    args = MagicMock()
    args.hostname = "example.com"

    mock_res = MagicMock()
    example_response = {"daily":{"time":["2023-01-01"],"weathercode":[1],"temperature_2m_max":[1],"temperature_2m_min":[1],"precipitation_sum":[1]}}
    mock_res.read.return_value = json.dumps(example_response).encode("utf-8")
    mock_conn_instance = mock_http_client.return_value
    mock_conn_instance.getresponse.return_value = mock_res
    
    # Mock the current date
    mock_date = datetime(2023, 1, 1)
    with patch('run.datetime') as mock_datetime, \
            patch('run.wandb.init'):
        mock_datetime.now.return_value = mock_date

        # Run the function
        go(args)

    # Check that the request method was called exactly 10 times
    assert mock_conn_instance.request.call_count == 10

    # Check that response is as expected
    mock_res.read.return_value = example_response

    # Ensure that the CSV file is saved with the correct name
    assert os.path.exists('data/raw_data.csv')
    assert os.path.exists('reports/ingested_data.txt')

