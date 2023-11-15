"""
This step takes the latest model and tests it against the test dataset.
If it performs better than previous models, it is promoted to a production.
"""
# pylint: disable=E0401, W0621, C0103, E1101, R0914, R0915
import os
import shutil
import logging
from datetime import datetime
import mlflow
import numpy as np
import wandb
import tempfile
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Set up logging
logging.basicConfig(
    filename=f"../{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(ARGS):
    """
    Test model perfromance and promote to production if better than previous models
    """
    LOGGER.info("6 - Running model testing step")

    run = wandb.init(project="weather-prediction",job_type="model_test")
    run.config.update(ARGS)

    LOGGER.info(
        "Downloading models- %s, %s and data- %s artifacts",
        ARGS.reg_model,
        ARGS.class_model,
        ARGS.test_dataset,
        ARGS.performance_records
    )
    # Downloading model artifact
    reg_model_local_path = run.use_artifact(ARGS.reg_model).download()
    class_model_local_path = run.use_artifact(ARGS.class_model).download()
    # Downloading test dataset
    test_dataset_path = run.use_artifact(ARGS.test_dataset).file()
    performance_records_path = run.use_artifact(ARGS.performance_records).file()

    df = pd.read_csv(test_dataset_path)

    df.set_index('time', inplace=True)

    LOGGER.info("Setting feature and target columns")
    reg_X_test = df.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
    reg_y_test = df[['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

    class_X_test = df.drop(['weathercode'], axis=1)
    class_y_test = df[['weathercode']]

    LOGGER.info("Loading model and performing inference on test set")
    reg_model = mlflow.sklearn.load_model(reg_model_local_path)
    class_model = mlflow.sklearn.load_model(class_model_local_path)
    reg_y_pred = reg_model.predict(reg_X_test)
    class_y_pred = class_model.predict(class_X_test)

    LOGGER.info("Scoring")
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
        # Do the inference and Compute the metrics
        slice_mae[val] = mean_absolute_error(reg_y_test[idx], reg_y_pred[idx])
        LOGGER.info("MAE of slices: %s", slice_mae[val])

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
    new = {'Date':date, 'Score':total_r_squared, 'MAE':total_mae}
    new = pd.DataFrame([new])
    new.set_index('Date', inplace=True)
    perf = pd.concat([perf,new])
    perf = perf.drop_duplicates()

    print(perf)

    LOGGER.info(
        "Checking if the new model is perfroming better than previous models")
    # If the MAE score of the latest model is smaller (better performace) than
    # any other models MAE, then this model is promoted to production model
    if total_mae <= perf['MAE'].min() or raw_comp or param_signific or nonparam:
        for n, name, model, temp_path in zip(['reg', 'class'],
                                     ['Regression','Classification'],
                                     [reg_model,class_model],
                                     [os.path.join(tempfile.gettempdir(), 'reg_model_dir'),
                                      os.path.join(tempfile.gettempdir(), "class_model_dir")]):
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path)
            mlflow.sklearn.save_model(model, temp_path)

            artifact = wandb.Artifact(
                f'{n}_model',
                type='model_export',
                description=f'Prodcution {name} model',
            )
            if not os.getenv('TESTING'):
                artifact.add_dir(temp_path)
                run.log_artifact(artifact, aliases=["prod"])
                artifact.wait()
            else:
                pass
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
    run.summary['reg_r2'] = reg_r_squared
    run.summary['reg_mae'] = reg_mae
    run.summary['class_r2'] = class_r_squared
    run.summary['class_mae'] = class_mae
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
        "--reg_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    PARSER.add_argument(
        "--class_model",
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

    PARSER.add_argument("--performance_records", type=str, help="Input artifact to split")

    ARGS = PARSER.parse_args()

    go(ARGS)