import pandas as pd
import joblib
from datetime import datetime, timedelta

df = pd.read_csv('./data/clean_data.csv')

recorded_data = df.loc['2023-10-12':'2023-10-18']

#predicted_data = pd.read_csv('./reports/next_week_prediction.csv')

#df_diff = recorded_data.compare(predicted_data)

#print(df_diff)

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
