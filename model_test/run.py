import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

rand = joblib.load("./training_validation/model_dir/rand.joblib")

df = pd.read_csv('./data/test.csv')
X_test = df.drop(['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'], axis=1)
y_test = df[['weathercode', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']]

X_test.set_index('time', inplace=True)
#y_test.set_index('time', inplace=True)

y_pred = rand.predict(X_test)

r_squared = rand.score(X_test, y_test)

mae = mean_absolute_error(y_test, y_pred)

print(r_squared)
print(mae)

"""slice_mae = {}
for val in y_test.unique():
    # Fix the feature
    idx = y_test == val

    # Do the inference and Compute the metrics
    preds = rand.predict(X_test[idx])
    slice_mae[val] = mean_absolute_error(y_test[idx], preds)"""

