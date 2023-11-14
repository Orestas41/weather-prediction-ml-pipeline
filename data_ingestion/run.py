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
    filename=f"../reports/logs/{date.strftime('%Y-%m-%d')}.log",
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
        data_save_path = '../data/raw_data.csv'
        report_save_path = "../reports/ingested_data.txt"
    else:
        config_path = 'config.yaml'
        # Use a temporary directory for testing
        if not os.path.exists('data'):
            os.makedirs('data')
        data_save_path = os.path.join(tempfile.gettempdir(), 'raw_data.csv')
        report_save_path = os.path.join(tempfile.gettempdir(), 'ingested_data.txt')    
    
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
    df_merged = pd.DataFrame()

    for i in city:
        LOGGER.info(f"Pulling weather data for {i}")
        conn.request("GET", f"/v1/archive?latitude={config['cities'][i]['latitude']}&longitude={config['cities'][i]['longitude']}&start_date={start.strftime('%Y-%m-%d')}&end_date={end.strftime('%Y-%m-%d')}&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = json.loads(data.decode("utf-8"))
        data = pd.DataFrame(data['daily'])
        data["city"] = config['cities'][i]['id']
        df_merged = pd.concat([df_merged, data])

    LOGGER.info("Saving merged data as csv file")
    df_merged.to_csv(data_save_path, index=False)

    LOGGER.info("Saving ingestion range record")
    data_record = open(
        report_save_path,"w")
    data_record.write(str(date.strftime('%Y-%m-%d')) + f' - data pulled from {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}' + '\n')

    LOGGER.info("Data ingestion finished")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument("--step_description", type=str,
                        help="Description of the step")
    
    PARSER.add_argument("--hostname", type=str,
        help="HTTPS connection to the server with the hostname",
        required=True
        )

    args = PARSER.parse_args()

    go(args)
