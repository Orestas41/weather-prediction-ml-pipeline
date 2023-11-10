import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import os
import sys

sys.path.append("/home/orestas41/weather-prediction-ml-pipeline/pre-processing")

# Import the 'go' function from run.py
from run import go, convert_date_column, create_month_day_column, set_date_index, merge_datasets, remove_na, drop_duplicates, sort_by_date

# Test the convert_date_column function
def test_convert_date_column():
    data = pd.DataFrame({'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    result = convert_date_column(data)
    assert 'time' in result.columns
    assert isinstance(result['time'][0], pd.Timestamp)

# Test the create_month_day_column function
def test_create_month_day_column():
    data = pd.DataFrame({'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    data = convert_date_column(data)
    result = create_month_day_column(data)
    assert 'month-day' in result.columns
    assert result['month-day'].dtype == int

# Test the set_date_index function
def test_set_date_index():
    data = pd.DataFrame({'time': ['2023-01-01', '2023-01-02'], 'value': [1, 2]})
    result = set_date_index(data)
    assert result.index.name == 'time'

# Test the merge_datasets function
def test_merge_datasets():
    data1 = pd.DataFrame({'time': ['2023-01-01'], 'value': [1]})
    data2 = pd.DataFrame({'time': ['2023-01-02'], 'value': [2]})
    result = merge_datasets(data1, data2)
    assert len(result) == 2

# Test the remove_na function
def test_remove_na():
    data = pd.DataFrame({'time': ['2023-01-01', '2023-01-02'], 'value': [1, None]})
    result = remove_na(data)
    assert len(result) == 1

# Test the drop_duplicates function
def test_drop_duplicates():
    data = pd.DataFrame({'time': ['2023-01-01', '2023-01-01'], 'value': [1, 1]})
    result = drop_duplicates(data)
    assert len(result) == 1

# Test the sort_by_date function
def test_sort_by_date():
    data = pd.DataFrame({'time': ['2023-01-02', '2023-01-01'], 'value': [2, 1]})
    data = convert_date_column(data)
    data = set_date_index(data)
    result = sort_by_date(data)
    #assert result.index[0] == '2023-01-01'
    assert result.index[0] == pd.Timestamp('2023-01-01')

# Test the go function
def test_go():
    os.environ['TESTING'] = '1'
    args = MagicMock()
    args.output_artifact = "output.csv"
    args.output_type = "type"
    args.output_description = "description"

    with patch('run.wandb.init'):
        go(args)