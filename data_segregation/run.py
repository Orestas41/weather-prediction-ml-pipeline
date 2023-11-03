import pandas as pd
from sklearn.model_selection import train_test_split

data_frame = pd.read_csv('./data/clean_data.csv')

trainval, test = train_test_split(
        data_frame,
        test_size=0.3,
    )

trainval.set_index('time', inplace=True)
test.set_index('time', inplace=True)

trainval.to_csv('./data/trainval.csv')
test.to_csv('./data/test.csv')


