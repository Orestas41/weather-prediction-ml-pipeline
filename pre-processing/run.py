import pandas as pd
import json

with open('./data/data.json') as f:
    data = json.load(f)
df = pd.DataFrame(data['daily'])
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
df['month-day'] = df['time'].dt.strftime('%m-%d')
df['month-day'] = pd.to_datetime(df['month-day'], format='%m-%d')
df['month-day'] = pd.to_datetime(df['month-day']).dt.strftime('%m%d').astype(int)

df.set_index('time', inplace=True)
df.to_csv("./data/clean_data.csv")