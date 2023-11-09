"""
This script splits the provided dataframe into a test set and a remainder set
"""
# pylint: disable=E0401, W0621, C0103, E1101
import tempfile
import logging
from datetime import datetime
import wandb
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Setting up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Splits the provided dataframe into test and remainder sets
    """

    run = wandb.init(project="weather-prediction",
                        job_type="data_segregation")
    run.config.update(ARGS)

    LOGGER.info("4 - Running data segregation step")

    LOGGER.info("Fetching artifact %s", ARGS.input)
    artifact_local_path = run.use_artifact(ARGS.input).file()

    data_frame = pd.read_csv(artifact_local_path)

    LOGGER.info("Splitting data into trainval and test")
    trainval, test = train_test_split(
        data_frame,
        test_size=ARGS.test_size,
    )

    trainval.set_index('time', inplace=True)
    test.set_index('time', inplace=True)

    #trainval.to_csv('../data/trainval.csv')
    #test.to_csv('../data/test.csv')

    for data_frame, k in zip([trainval, test], ['trainval', 'test']):
        LOGGER.info("Uploading %s_data.csv dataset", k)
        with tempfile.NamedTemporaryFile("w") as file:
            data_frame.to_csv(file.name, index=True)
            artifact = wandb.Artifact(
                f"{k}_data.csv",
                type=f"{k}_data",
                description=f"{k} split of dataset",
            )
            artifact.add_file(file.name)
            run.log_artifact(artifact)
            artifact.wait()

    LOGGER.info("Finished data segregation")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Split test and remainder")

    PARSER.add_argument("input", type=str, help="Input artifact to split")

    PARSER.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")

    ARGS = PARSER.parse_args()

    go(ARGS)