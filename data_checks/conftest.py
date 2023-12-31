"""
This script defines shared fixtures and configuration options for test files
"""
# pylint: disable=E1101, E0401
import pytest
import pandas as pd
import wandb

def pytest_addoption(parser):
    """
    Add command-line options.
    """
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store", type=float)


@pytest.fixture(scope='session')
def data(request):
    """
    Read the data from a CSV file.
    """
    run = wandb.init(job_type="data_check", resume=True)
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")
    data_frame = pd.read_csv(data_path)
    return data_frame


@pytest.fixture(scope='session')
def ref_data(request):
    """
    Read the reference data from a CSV file.
    """
    run = wandb.init(job_type="data_check", resume=True)
    data_path = run.use_artifact(request.config.option.ref).file()
    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")
    data_frame = pd.read_csv(data_path)
    return data_frame


@pytest.fixture(scope='session')
def threshold(request):
    """
    Get the KL threshold from the command line.
    """
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")
    return kl_threshold
