"""
Evaluating previous tour results against models predictions.
Batch predicting next tour match results
"""
# pylint: disable=E0401, W0621, R0914, E1101, C0200, C0103, W0106, R0915
"""import csv
import os.path
import logging
import json
import argparse
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import mlflow
import wandb"""
import joblib
from datetime import datetime
import pandas as pd

# Set up logging
"""logging.basicConfig(
    filename=f"../reports/logs/{datetime.now().strftime('%Y-%m-%d')}.log",
    level=logging.INFO)
LOGGER = logging.getLogger()"""


#def go(ARGS):
"""
Evaluating previous tour results against models predictions.
Batch predicting next tour match results
"""
"""LOGGER.info("7 - Running tour evaluation and prediction step")

run = wandb.init(
    job_type="data_scraping")
run.config.update(ARGS)

LOGGER.info("Configuring webdriver")
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")

homedir = os.path.expanduser("~")
webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")

LOGGER.info("Setting browser")
with webdriver.Chrome(service=webdriver_service, options=chrome_options) as driver:

result_website = "https://alyga.lt/rezultatai/1"
LOGGER.info(
    "Opening website for of the latest results - %s", result_website)
driver.get(result_website)

LOGGER.info("Scraping the data")
rows = driver.find_elements(By.TAG_NAME, "tr")

# Saving the data to results csv file
with open(f"../reports/tours/result.csv", 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the data rows
    for row in rows[1:6]:
        data = row.find_elements(By.TAG_NAME, "td")
        writer.writerow([datum.text for datum in data])

LOGGER.info("Inserting the data to dataframe")
data_frame = pd.read_csv("../reports/tours/result.csv", header=None)

LOGGER.info("Applying pre-processing")
data_frame.columns = ["Date", "Blank",
                        "Home", "Result", "Away", "Location"]

# Convert dates to datetime format and timestamps
data_frame['Date'] = pd.to_datetime(
    data_frame['Date'], format='%Y-%m-%d, %H:%M').astype(int) / 10**18

# Convert Result column into separate columns for Home and Away goals
data_frame[['Home Result', 'Away Result']] = data_frame['Result'].str.split(
    ' : ', expand=True).astype(int)

# Create Winner column with the team that won or draw
data_frame['Winner'] = np.where(
    data_frame['Home Result'] > data_frame['Away Result'], 0, np.where(
        data_frame['Home Result'] < data_frame['Away Result'], 1, 0.5))

# Removing uneccessary columns
data_frame = data_frame.drop(['Blank', 'Location', 'Result',
                                'Home Result', 'Away Result'], axis=1)

LOGGER.info("Adding previous predictions for this tour")
with open('../reports/tours/predictions.json') as file:
    prev_preds = pd.DataFrame(json.load(file))

data_frame = pd.merge(data_frame, prev_preds, on='Date', how='outer')

LOGGER.info("Calculating prediction error")
data_frame['Model Performance'] = abs(
    data_frame['Winner'] - data_frame['Prediction'])

LOGGER.info(
    "Saving the report on the latest tour prediction evaluations")
data_frame.to_csv(
    f"../reports/tours/{datetime.now().strftime('%Y-%m-%d')}.csv",
    index=None)

fixture_website = "https://alyga.lt/tvarkarastis/1"
LOGGER.info(
    "Opening website of the future tour fixtures - %s", fixture_website)
driver.get(fixture_website)

LOGGER.info("Scraping the data")
rows = driver.find_elements(By.TAG_NAME, "tr")

# Saving the data to next_tour csv file
with open(f"../reports/tours/next_tour.csv", 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the data rows
    for row in rows[1:6]:
        data = row.find_elements(By.TAG_NAME, "td")
        writer.writerow([datum.text for datum in data])

LOGGER.info("Inserting the data into a dataframe")
data_frame1 = pd.read_csv(
    "../reports/tours/next_tour.csv", header=None)

LOGGER.info("Applying pre-processing")
data_frame1.columns = ["Date", "Blank",
                        "Home", "TV", "Away", "Location"]
data_frame1['Date'] = pd.to_datetime(
    data_frame1['Date'], format='%Y-%m-%d, %H:%M').astype(int) / 10**18

LOGGER.info("Loading the encoder")
with open('../pre-processing/encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

home = data_frame1['Home']
away = data_frame1['Away']

LOGGER.info("Encoding home and away team names")
data_frame1['Home'] = encoder.transform(data_frame1['Home'])
data_frame1['Away'] = encoder.transform(data_frame1['Away'])

# Removing unecessary
data_frame1 = data_frame1.drop(['Blank', 'Location', 'TV'], axis=1)

LOGGER.info(
    "Loading the model and predicting the results for next tour fixtures")
dirname = os.path.dirname(__file__)
model = mlflow.sklearn.load_model(os.path.join(
    dirname, "../prod_model_dir"))
pred = model.predict(data_frame1)

preds_json = dict(
    Date=data_frame1["Date"].tolist(), Prediction=pred.tolist())

LOGGER.info("Saving the predictions to json file")
with open('../reports/tours/predictions.json', 'w') as file:
    json.dump(preds_json, file, indent=4)

LOGGER.info("Writing models predictions to a presentable txt file")
with open("../reports/tours/predictions_for_next_tour.txt", "w") as f:
    for i in range(len(pred)):
        f.write(
            f"For the match between {home[i]} and {away[i]}, model predicts that"
            + f" {home[i]}'s chance of winning is {(pred[i]*100).astype(int)}%.\n")

LOGGER.info("Removing temporary files")
os.remove("../reports/tours/next_tour.csv")
os.remove("../reports/tours/result.csv")

LOGGER.info("Batch tour evaluations and predictions finished")
driver.close()"""

df = pd.read_csv('./data/clean_data.csv')

recorded_data = df.loc['2023-10-12':'2023-10-18']

predicted_data = pd.read_csv('./reports/next_week_prediction.csv')

df_diff = recorded_data.compare(predicted_data)

print(df_diff)

# Create a date range for the next 7 days
date_rng = pd.date_range(start=datetime.now(), end=datetime.now() + timedelta(days=7), freq='D')

# Create a DataFrame with a date column
df = pd.DataFrame(date_rng, columns=['time'])
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
df['month-day'] = df['time'].dt.strftime('%m-%d')
df['month-day'] = pd.to_datetime(df['month-day'], format='%m-%d')
df['month-day'] = pd.to_datetime(df['month-day']).dt.strftime('%m%d').astype(int)
df.set_index('time', inplace=True)

rand = joblib.load("./training_validation/model_dir/rand.joblib")

preds = rand.predict(df)

df['weathercode'] = 0
df['temperature_2m_max'] = 0
df['temperature_2m_min'] = 0
df['precipitation_sum'] = 0

for i in range(len(preds)):
    df['weathercode'][i] = preds[i][0]
    df['temperature_2m_max'][i] = preds[i][1]
    df['temperature_2m_min'][i] = preds[i][2]
    df['precipitation_sum'][i] = preds[i][3]

df.to_csv("./reports/next_week_prediction.csv")


"""if __name__ == "__main__":

    PARSER = argparse.ArgumentParser(
        description="This step scrapes the latest data from the web")

    PARSER.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    ARGS = PARSER.parse_args()

    go(ARGS)"""