import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
import sys

sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/data_checks")

# Import the module containing the fixtures and functions to be tested
from conftest import pytest_addoption, data, ref_data, threshold

# Define a test fixture for the command line options
@pytest.fixture
def parser():
    class Parser:
        def addoption(self, *args, **kwargs):
            pass
    return Parser()

# Test for pytest_addoption function
def test_pytest_addoption(parser):
    pytest_addoption(parser)
    # Add your assertions here based on the expected behavior of pytest_addoption

# Test for the 'data' fixture
@pytest.fixture
def mock_wandb_init(monkeypatch):
    monkeypatch.setattr("conftest.wandb.init", MagicMock(return_value=MagicMock(file=MagicMock())))

"""@pytest.fixture
def data(monkeypatch):
    monkeypatch.setattr("data/training_data.csv")

@pytest.fixture
def ref_data(monkeypatch):
    monkeypatch.setattr("data/training_data.csv")

@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setattr(0.5)"""

@pytest.mark.usefixtures("mock_wandb_init")
def test_data_fixture(data):
    # Set up mock objects
    #request.config.option.csv = "data/training_data.csv"
    
    # Call the fixture
    #result = data(data)
    
    # Add assertions based on the expected behavior of the 'data' fixture
    assert isinstance(data, pd.DataFrame)
    # Add more assertions as needed

# Test for the 'ref_data' fixture
@pytest.mark.usefixtures("mock_wandb_init")
def test_ref_data_fixture(ref_data):
    # Set up mock objects
    #request.config.option.ref = "data/training_data.csv"
    
    # Call the fixture
    #result = ref_data(ref_data)
    
    # Add assertions based on the expected behavior of the 'ref_data' fixture
    assert isinstance(ref_data, pd.DataFrame)
    # Add more assertions as needed

# Test for the 'threshold' fixture
def test_threshold_fixture(threshold):
    # Set up mock objects
    #request.config.option.kl_threshold = 0.5
    
    # Call the fixture
    #result = threshold(request)
    
    # Add assertions based on the expected behavior of the 'threshold' fixture
    assert isinstance(threshold, float)
    assert threshold == 0.5  # Adjust based on your expected threshold
    # Add more assertions as needed
