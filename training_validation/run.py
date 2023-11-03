import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rand = RandomForestRegressor()

df = pd.read_csv('./data/trainval.csv')
X = df.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
y = df[['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

X_train.set_index('time', inplace=True)
X_val.set_index('time', inplace=True)

rand.fit(X_train, y_train)

r_squared = rand.score(X_val, y_val)
y_pred = rand.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)

if not os.path.exists("training_validation/model_dir"):
    os.makedirs("training_validation/model_dir")

joblib.dump(rand, "training_validation/model_dir/rand.joblib")

