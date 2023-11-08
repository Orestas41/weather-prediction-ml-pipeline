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
import yaml
from datetime import datetime, timedelta

# Setting up logging
logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()


def go(args):
    """
    Pulls weather data for each city from API, merges and saves it in a CSV file
    """
    LOGGER.info("1 - Running data ingestion step")
    run = wandb.init(job_type="data_ingestion")
    run.config.update(args)
    
    # Opening configuration file
    with open('../config.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

    # Setting up API request
    hostname = run.use_artifact(args.hostname).file()

    conn = http.client.HTTPSConnection(hostname)
    payload = ''
    headers = {}

    city = config['cities']
    start = datetime.now() - timedelta(days=8)
    end = datetime.now() - timedelta(days=1)
    df_merged = pd.DataFrame()

    for i in city:
        LOGGER.info(f"Pulling weather data for {i}")
        conn.request("GET", f"/v1/archive?latitude={config['cities'][i]['latitude']}&longitude={config['cities'][i]['longitude']}&start_date={start}&end_date={end}&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = json.loads(data.decode("utf-8"))
        data["city"] = config['cities'][i]['id']
        data = pd.DataFrame(data['daily'])
        data['city'] = data['city']
        df_merged = pd.concat([df_merged, data])

    LOGGER.info("Saving merged data as csv file")
    df_merged.to_csv(f'../data/raw_data.csv')

    # Recording data range pulled
    data_record = open(
        f"../reports/ingested_data.txt","w")
    data_record.write(str(datetime.now().strftime('%Y-%m-%d')) + f' - data pulled from {start} to {end}' + '\n')

    LOGGER.info("Data ingestion finished")


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument("--step_description", type=str,
                        help="Description of the step")
    
    PARSER.add_argument(
        "hostname",
        type=str,
        help="HTTPS connection to the server with the hostname")

    args = PARSER.parse_args()

    go(args)
