"""
This step takes the latest model and tests it against the test dataset.
If it performs better than previous models, it is promoted as a production model.
"""
# pylint: disable=E0401, W0621, C0103, E1101, R0914, R0915
import os
import shutil
import logging
from datetime import datetime
import tempfile
import argparse
import numpy as np
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import wandb

# Set up logging
logging.basicConfig(
    filename=f"../{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Test model performance and promote to production if better than previous models
    """
    LOGGER.info("5 - Running model testing step")

    run = wandb.init(job_type="model_test")
    run.config.update(ARGS)

    LOGGER.info(
        "Downloading %s and %s artifacts",
        ARGS.test_dataset,
        ARGS.performance_records
    )
    # Downloading test dataset
    test_dataset_path = run.use_artifact(ARGS.test_dataset).file()
    performance_records_path = run.use_artifact(
        ARGS.performance_records).file()

    # Setting the file path of temporary stored models
    reg_model_local_path = '../reg_model_dir'
    class_model_local_path = '../class_model_dir'

    LOGGER.info("Opening test dataset")
    df = pd.read_csv(test_dataset_path)

    # Setting time column as index
    df.set_index('time', inplace=True)

    LOGGER.info("Setting feature and target columns")
    reg_X_test = df.drop(['weathercode',
                          'temperature_2m_max',
                          'temperature_2m_min',
                          'precipitation_sum'],
                         axis=1)
    reg_y_test = df[['temperature_2m_max',
                     'temperature_2m_min', 'precipitation_sum']]

    class_X_test = df.drop(['weathercode'], axis=1)
    class_y_test = df[['weathercode']]

    LOGGER.info("Loading the models and performing inference on the test sets")
    reg_model = mlflow.sklearn.load_model(reg_model_local_path)
    class_model = mlflow.sklearn.load_model(class_model_local_path)

    reg_y_pred = reg_model.predict(reg_X_test)
    class_y_pred = class_model.predict(class_X_test)

    LOGGER.info("Scoring both models")
    reg_r_squared = reg_model.score(reg_X_test, reg_y_test)
    class_r_squared = class_model.score(class_X_test, class_y_test)
    total_r_squared = (reg_r_squared + class_r_squared) / 2
    LOGGER.info("Regression Score: %s", reg_r_squared)
    LOGGER.info("Classification Score: %s", class_r_squared)
    LOGGER.info("Total Score: %s", total_r_squared)

    reg_mae = mean_absolute_error(reg_y_test, reg_y_pred)
    class_mae = mean_absolute_error(class_y_test, class_y_pred)
    total_mae = (reg_mae + class_mae) / 2
    LOGGER.info("Regression MAE: %s", reg_mae)
    LOGGER.info("Classification MAE: %s", class_mae)
    LOGGER.info("Total MAE: %s", total_mae)

    LOGGER.info("Running data slice tests")
    slice_mae = {}
    for val in reg_X_test['city'].unique():
        idx = reg_X_test['city'] == val
        slice_mae[val] = mean_absolute_error(reg_y_test[idx], reg_y_pred[idx])
        LOGGER.info("MAE score of %s slice: %s", val, slice_mae[val])

    LOGGER.info("Testing data drift. Expecting results to be False")
    # Opening model performance log
    perf = pd.read_csv(performance_records_path, index_col=0)

    # Raw comparison test
    raw_comp = total_r_squared < np.min(perf['Score'])
    LOGGER.info("Raw comparison: %s", raw_comp)

    # Parametric significance test
    param_signific = total_r_squared < np.mean(
        perf['Score']) - 2 * np.std(perf['Score'])
    LOGGER.info("Parametric significance: %s", param_signific)

    # Non-parametric outlier test
    iqr = np.quantile(perf['Score'], 0.75) - np.quantile(perf['Score'], 0.25)
    nonparam = total_r_squared < np.quantile(perf['Score'], 0.25) - iqr * 1.5
    LOGGER.info("Non-parametric outlier: %s", nonparam)

    LOGGER.info(
        "Saving the latest model performance metrics")
    date = datetime.now().strftime('%Y-%m-%d')
    new = {'Date': date, 'Score': total_r_squared, 'MAE': total_mae}
    new = pd.DataFrame([new])
    new.set_index('Date', inplace=True)
    perf = pd.concat([perf, new])
    perf = perf.drop_duplicates()
    LOGGER.info("\n%s", perf.to_string(index=True))

    LOGGER.info(
        "Checking if the new model is performing better than previous models")
    # If the MAE score of the latest model is smaller (better performance) than
    # any other models MAE, then this model is promoted as a production model
    if total_mae <= perf['MAE'].min(
    ) or raw_comp or param_signific or nonparam:
        alias = ['latest', 'prod']
        LOGGER.info(
            "Model has the best performance. Uploading as the production model")
    else:
        alias = ['latest']
        LOGGER.info(
            "Model did not perform better than production model. Uploading as the latest model")

    for n, name, path in zip(['reg', 'class'],
                             ['Regression', 'Classification'],
                             [reg_model_local_path, class_model_local_path]):
        artifact = wandb.Artifact(
            f'{n}_model',
            type='model_export',
            description=f'Prodcution {name} model',
        )
        if not os.getenv('TESTING'):
            artifact.add_dir(path)
            run.log_artifact(artifact, aliases=alias)
            artifact.wait()
        else:
            pass

    with tempfile.NamedTemporaryFile("w") as file:
        perf.to_csv(file.name, index=True)
        LOGGER.info("Uploading performance records")
        artifact = wandb.Artifact(
            'model_performance.csv',
            type='performance_records',
            description='performance_records',
        )
        artifact.add_file(file.name)
        run.log_artifact(artifact)
        if not os.getenv('TESTING'):
            artifact.wait()
        else:
            pass

    # Logging MAE and r2
    for name, metric in zip(['reg_mae', 'class_mae', 'total_mae',
                             'reg_r2', 'class_r2', 'total_r2',
                             'Raw comparison', 'Parametric significance', 'Non-parametric outlier'],
                            [reg_mae, class_mae, total_mae,
                             reg_r_squared, class_r_squared, total_r_squared,
                             raw_comp, param_signific, nonparam]):
        run.summary[name] = metric

    LOGGER.info("Model testing finished")

    # Removing locally saved models
    if not os.getenv('TESTING'):
        for path in ['../reg_model_dir', '../class_model_dir']:
            shutil.rmtree(path)
    else:
        pass

    # Finish the run
    run.finish()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="Test the provided model against the test dataset")

    PARSER.add_argument(
        "--reg_model",
        type=str,
        help="Input MLFlow Regression model",
        required=True
    )

    PARSER.add_argument(
        "--class_model",
        type=str,
        help="Input MLFlow Classification model",
        required=True
    )

    PARSER.add_argument(
        "--test_dataset",
        type=str,
        help="Input test dataset",
        required=True
    )

    PARSER.add_argument(
        "--performance_records",
        type=str,
        help="Input the csv file with records of previous model performances",
        required=True
    )

    ARGS = PARSER.parse_args()

    go(ARGS)
