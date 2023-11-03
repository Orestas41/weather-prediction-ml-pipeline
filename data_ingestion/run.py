"""
This script pulls the latest data from the api
"""
# pylint: disable=E1101, E0401, C0103, W0621
"""import csv
import os.path
import logging
import argparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import wandb"""
import http.client
import json

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

LOGGER.info("Configuring webdriver")
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")

homedir = os.path.expanduser("~")
webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")

LOGGER.info("Setting browser")
with webdriver.Chrome(service=webdriver_service, options=chrome_options) as driver:
    website = "https://alyga.lt/rezultatai/1"
    LOGGER.info("Opening %s", website)
    driver.get(website)

    LOGGER.info("Scraping the data")
    rows = driver.find_elements(By.TAG_NAME, "tr")

    # Opening csv file with today's date as name
    with open(f"../raw_data/{datetime.now().strftime('%Y-%m-%d')}.csv", 'w', newline='')as file:
        writer = csv.writer(file)

        # Write the data rows
        for row in rows[1:]:
            data = row.find_elements(By.TAG_NAME, "td")
            writer.writerow([datum.text for datum in data])

    LOGGER.info("Scraping finished")
    driver.close()"""

conn = http.client.HTTPSConnection("archive-api.open-meteo.com")
payload = ''
headers = {}
conn.request("GET", f"/v1/archive?latitude=51.45&longitude=-2.58&start_date=2023-05-18&end_date=2023-10-20&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=Europe%2FLondon", payload, headers)
res = conn.getresponse()
data = res.read()
data = json.loads(data.decode("utf-8"))
with open("./data/data.json", "w") as write_file:
    json.dump(data, write_file)

"""if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument("--step_description", type=str,
                        help="Description of the step")

    args = PARSER.parse_args()

    go(args)"""
