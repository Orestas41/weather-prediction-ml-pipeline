"""
This script pulls the latest data from the api
"""
# pylint: disable=E1101, E0401, C0103, W0621
"""import csv
import os.path
import logging
import argparse
import wandb"""
import http.client
import json
import yaml
from datetime import datetime

# Setting up logging
"""logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()"""


#def go(args):
"""
Scrapes data from a website and saves it in a CSV file
"""
"""run = wandb.init(job_type="data_scraping")
run.config.update(args)
LOGGER.info("1 - Running data scrape step")
"""

with open('./config.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

conn = http.client.HTTPSConnection("archive-api.open-meteo.com")
payload = ''
headers = {}

city = config['cities']

for i in city:
    conn.request("GET", f"/v1/archive?latitude={config['cities'][i]['latitude']}&longitude={config['cities'][i]['longitude']}&start_date=2023-05-18&end_date=2023-10-20&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
    res = conn.getresponse()
    data = res.read()
    data = json.loads(data.decode("utf-8"))
    data["city"] = config['cities'][i]['id']
    with open(f"./data/raw/{i}_data.json", "w") as write_file:
        json.dump(data, write_file)


"""
    LOGGER.info("Scraping finished")
    driver.close()"""



"""if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument("--step_description", type=str,
                        help="Description of the step")

    args = PARSER.parse_args()

    go(args)"""
