"""
This script pulls the latest data from the api
"""
# pylint: disable=E1101, E0401, C0103, W0621
import logging
import wandb
import argparse
import http.client
import pandas as pd
import json
import os
import tempfile
import yaml
from datetime import datetime, timedelta

date = datetime.now()

# Setting up logging
logging.basicConfig(
    filename=f"../{date.strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()

def go(args):
    """
    Pulls weather data for each city from API, merges and saves it in a CSV file
    """
    LOGGER.info("1 - Running data ingestion step")
    run = wandb.init(job_type="data_ingestion")
    run.config.update(args)

    LOGGER.info("Setting up file locations according to the environment")
    if not os.getenv('TESTING'):
        config_path = '../config.yaml'
    else:
        config_path = 'config.yaml'

    LOGGER.info("Fetching %s artifact", args.ingestion_records)
    ingestion_records_path = run.use_artifact(args.ingestion_records).file()

    # Opening configuration file
    with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    # Setting up API request
    conn = http.client.HTTPSConnection(args.hostname)
    payload = ''
    headers = {}

    city = config['cities']
    end = date - timedelta(days=7)
    start = end - timedelta(days=7)
    raw_data = pd.DataFrame()

    for i in city:
        LOGGER.info(f"Pulling weather data for {i}")
        conn.request("GET", f"/v1/archive?latitude={config['cities'][i]['latitude']}&longitude={config['cities'][i]['longitude']}&start_date={start.strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = json.loads(data.decode("utf-8"))
        data = pd.DataFrame(data['daily'])
        data["city"] = config['cities'][i]['id']
        raw_data = pd.concat([raw_data, data])

    LOGGER.info("Saving ingestion range record")
    ingestion_records = pd.read_csv(ingestion_records_path)
    ingestion_records = ingestion_records.assign(Date= [date.strftime('%Y-%m-%d')])
    ingestion_records = ingestion_records.assign(Start= [start.strftime('%Y-%m-%d')])
    ingestion_records = ingestion_records.assign(End= [end.strftime('%Y-%m-%d')])
    
    for file_name, k, desc in zip([raw_data,ingestion_records],
                                  ['raw_data.csv', 'ingestion_records.csv'],
                                  ['raw_data','ingestion_records']):
        LOGGER.info("Uploading %s", desc)
        with tempfile.NamedTemporaryFile("w") as file:
            file_name.to_csv(file.name, index=False)
            artifact = wandb.Artifact(
                k,
                type=desc,
                description=desc,
            )
            artifact.add_file(file.name)
            run.log_artifact(artifact)
            if not os.getenv('TESTING'):
                artifact.wait()
            else:
                pass

    LOGGER.info("Data ingestion finished")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")
    
    PARSER.add_argument("--ingestion_records", type=str, help="Input artifact to split")

    PARSER.add_argument("--step_description", type=str,
                        help="Description of the step")
    
    PARSER.add_argument("--hostname", type=str,
        help="HTTPS connection to the server with the hostname",
        required=True
        )

    args = PARSER.parse_args()

    go(args)
