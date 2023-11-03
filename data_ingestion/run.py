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
conn = http.client.HTTPSConnection("archive-api.open-meteo.com")
payload = ''
headers = {}
conn.request("GET", f"/v1/archive?latitude=51.45&longitude=-2.58&start_date=2023-05-18&end_date=2023-10-20&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
res = conn.getresponse()
data = res.read()
data = json.loads(data.decode("utf-8"))
with open(f"./data/new_data.json", "w") as write_file:
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
