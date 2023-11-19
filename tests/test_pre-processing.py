"""
This test script tests functions from pre-processing step
"""
# pylint: disable=C0103, E0401, C0413, W0621
import os
import sys
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Add the path to the directory containing the script you want to test
main_path = os.getcwd()
sys.path.append(
    f"{main_path}/pre-processing")

# Import the functions to be tested
from run import (go, convert_date_column,
                 create_month_day_column, set_date_index,
                 merge_and_clean_datasets, sort_by_date)


def test_convert_date_column():
    """Test the convert_date_column function"""
    data = pd.DataFrame(
        {'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    result = convert_date_column(data)
    assert 'time' in result.columns
    assert isinstance(result['time'][0], pd.Timestamp)


def test_create_month_day_column():
    """Test the create_month_day_column function"""
    data = pd.DataFrame(
        {'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    data = convert_date_column(data)
    result = create_month_day_column(data)
    assert 'month-day' in result.columns
    assert result['month-day'].dtype == float


def test_set_date_index():
    """Test the set_date_index function"""
    data = pd.DataFrame(
        {'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    result = set_date_index(data)
    assert result.index.name == 'time'


def test_merge_and_clean_datasets():
    """Test the merge_and_clean_datasets function"""
    data1 = pd.DataFrame(
        {'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    data2 = pd.DataFrame(
        {'time': ['2023-01-01', '2023-01-02'], 'value': [None, 2]})
    result = merge_and_clean_datasets(data1, data2)
    assert len(result) == 2


def test_sort_by_date():
    """Test the sort_by_date function"""
    data = pd.DataFrame(
        {'time': ['2023-01-02', '2023-01-01'], 'value': [2, 1]})
    data = convert_date_column(data)
    data = set_date_index(data)
    result = sort_by_date(data)
    assert result.index[0] == pd.Timestamp('2023-01-01')


@pytest.fixture
def mock_wandb():
    """Mock wandb connection"""
    with patch('run.wandb.init') as mock_wandb_init:
        yield mock_wandb_init

# Test the go function


def test_go(mock_wandb):
    """Test go function"""
    # Setting up test environment
    os.environ['TESTING'] = '1'

    # Setting up arguments
    args = MagicMock()
    args.raw_data = "tests/mock_data.csv"
    args.training_data = "tests/mock_data.csv"
    args.output_artifact = 'example.csv'
    args.output_type = "type"
    args.output_description = "description"

    # Mock wandb run
    mock_run = MagicMock()
    mock_wandb.return_value = mock_run

    # Mock test data
    mock_artifact = MagicMock()
    mock_artifact.file.return_value = 'tests/mock_data.csv'
    mock_run.use_artifact.return_value = mock_artifact

    # Run the function
    go(args)
