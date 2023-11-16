"""
This script represents a pipeline for machine learning tasks,
where each step can be executed independently based on the provided configuration
"""
# pylint: disable=E0401, C0103, E1120
import json
import tempfile
import os
import mlflow
import hydra
from omegaconf import DictConfig

_steps = [
    "data_ingestion",
    "pre-processing",
    "data_checks",
    "data_segregation",
    "training_validation",
    "model_test",
    "batch_prediction"
]


# Reading the configuration
@hydra.main(config_name='config.yaml', version_base='1.1')
def go(config: DictConfig):
    """
    Run the pipeline for machine learning tasks
    """

    # Setting-up the wandb experiment
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Moving to a temporary directory
    with tempfile.TemporaryDirectory():

        if "data_ingestion" in active_steps:
            
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data_ingestion"),
                "main",
                parameters={
                    "ingestion_records": "ingestion_records.csv:latest",
                    "step_description": "This step pull the latest weather data from API",
                    "hostname": config["data_ingestion"]["hostname"]
                    },
                    
            )

        if "pre-processing" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "pre-processing"),
                "main",
                parameters={
                    'raw_data':'raw_data.csv:latest',
                    'training_data':'training_data.csv:latest',
                    "output_artifact": "training_data.csv",
                    "output_type": "training_data",
                    "output_description": "New data merged with previous training data"
                },
            )

        if "data_checks" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data_checks"),
                "main",
                parameters={
                    "csv": "training_data.csv:latest",
                    "ref": "training_data.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"]
                    }
            )

        if "data_segregation" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "data_segregation"),
                "main",
                parameters={
                    "input": "training_data.csv:latest",
                    "test_size": config["data_segregation"]["test_size"]}
            )

        if "training_validation" in active_steps:

            reg_config = os.path.abspath("reg_config.yaml")
            with open(reg_config, "w+") as reg_file:
                json.dump(
                    dict(config["modeling"]["XGBRegressor"].items()), reg_file)
            class_config = os.path.abspath("class_config.yaml")
            with open(class_config, "w+") as class_file:
                json.dump(
                    dict(config["modeling"]["RandomForestClassifier"].items()), class_file)
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "training_validation"),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "reg_config": reg_config,
                    "class_config": class_config},
            )

        if "model_test" in active_steps:

            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "model_test"),
                "main",
                parameters={
                    "reg_model": "reg_model:latest",
                    "class_model": "class_model:latest",
                    "test_dataset": "test_data.csv:latest",
                    "performance_records": "model_performance.csv:latest"},
            )

        if "batch_prediction" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "batch_prediction"),
                "main",
                parameters={
                    "reg_model": "reg_model:prod",
                    "class_model": "class_model:prod",
                    "full_dataset": "training_data.csv:latest",
                    "batch_prediction": "batch_prediction.csv:latest"
                },
            )


if __name__ == "__main__":
    go()