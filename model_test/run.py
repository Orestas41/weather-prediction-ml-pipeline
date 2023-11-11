"""
This step takes the latest model and tests it against the test dataset.
If it performs better than previous models, it is promoted to a production.
"""
# pylint: disable=E0401, W0621, C0103, E1101, R0914, R0915
import os
import csv
import shutil
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import wandb
import tempfile
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Set up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Test model perfromance and promote to production if better than previous models
    """
    LOGGER.info("6 - Running model testing step")

    LOGGER.info("Setting up file locations according to the environment")
    if not os.getenv('TESTING'):
        performance_report_path = '../reports/model_performance.csv'
        performance_report_save_path = '../reports/model_performance.csv'
        prod_model_path = "../prod_model_dir"
        performance_plot_path = "../reports/model_performance.png"
    else:
        performance_report_path = 'reports/model_performance.csv'
        # Use a temporary directory for testing
        if not os.path.exists('data'):
            os.makedirs('data')
        performance_report_save_path = os.path.join(tempfile.gettempdir(), 'model_performance.csv')
        prod_model_path = os.path.join(tempfile.gettempdir(), "prod_model_dir")
        performance_plot_path = os.path.join(tempfile.gettempdir(), "model_performance.png")

    run = wandb.init(job_type="model_test")
    run.config.update(ARGS)

    LOGGER.info(
        "Downloading model- %s and data- %s artifacts",
        ARGS.mlflow_model,
        ARGS.test_dataset
    )
    # Downloading model artifact
    model_local_path = run.use_artifact(ARGS.mlflow_model).download()
    # Downloading test dataset
    test_dataset_path = run.use_artifact(ARGS.test_dataset).file()

    df = pd.read_csv(test_dataset_path)

    LOGGER.info("Setting feature and target columns")
    X_test = df.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
    y_test = df[['time','weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

    X_test.set_index('time', inplace=True)
    y_test.set_index('time', inplace=True)

    LOGGER.info("Loading model and performing inference on test set")
    model = mlflow.sklearn.load_model(model_local_path)
    y_pred = model.predict(X_test)

    LOGGER.info("Scoring")
    r_squared = model.score(X_test, y_test)
    LOGGER.info("Score: %s", r_squared)
    mae = mean_absolute_error(y_test, y_pred)
    LOGGER.info("MAE: %s", mae)

    LOGGER.info("Running data slice tests")
    slice_mae = {}
    for val in X_test['city'].unique():
        idx = X_test['city'] == val
        # Do the inference and Compute the metrics
        slice_mae[val] = mean_absolute_error(y_test[idx], y_pred[idx])
        LOGGER.info("MAE of slices: %s", slice_mae[val])

    LOGGER.info("Testing data drift. Expecting results to be False")
    # Opening model performance log
    perf = pd.read_csv(performance_report_path, index_col=0)

    # Raw comparison test
    raw_comp = r_squared < np.min(perf['Score'])
    LOGGER.info("Raw comparison: %s", raw_comp)

    # Parametric significance test
    param_signific = r_squared < np.mean(
        perf['Score']) - 2 * np.std(perf['Score'])
    LOGGER.info("Parametric significance: %s", param_signific)

    # Non-parametric outlier test
    iqr = np.quantile(perf['Score'], 0.75) - np.quantile(perf['Score'], 0.25)
    nonparam = r_squared < np.quantile(perf['Score'], 0.25) - iqr * 1.5
    LOGGER.info("Non-parametric outlier: %s", nonparam)

    LOGGER.info(
        "Saving the latest model performance metrics")
    date = datetime.now().strftime('%Y-%m-%d')
    with open(performance_report_save_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([date, r_squared, mae])

    LOGGER.info(
        "Checking if the new model is perfroming better than previous models")
    # If the MAE score of the latest model is smaller (better performace) than
    # any other models MAE, then this model is promoted to production model
    if mae <= perf['MAE'].min() or raw_comp or param_signific or nonparam:
        if os.path.exists(prod_model_path):
            shutil.rmtree(prod_model_path)
        LOGGER.info("Model performance is better than previous model. Promoting new model to production")
        mlflow.sklearn.save_model(model, prod_model_path)
    else:
        pass

    
    LOGGER.info(
        "Generating plot that displays the change in model performance over time")
    performance = pd.read_csv(performance_report_path)
    plt.plot(performance["Date"], performance["Score"], label="Score")
    plt.plot(performance["Date"], performance["MAE"], label="MAE")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Score/MAE")
    plt.title("Change in ML Model Performance")

    # Save the plot
    plt.savefig(performance_plot_path)

    # Logging MAE and r2
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    run.summary["Raw comparison"] = raw_comp
    run.summary["Parametric significance"] = param_signific
    run.summary["Non-parametric outlier"] = nonparam

    LOGGER.info("Finished testing the model")

    # Finish the run
    run.finish()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="Test the provided model against the test dataset")

    PARSER.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    PARSER.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    ARGS = PARSER.parse_args()

    go(ARGS)