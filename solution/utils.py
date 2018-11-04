import datetime
import numpy as np


class ModelsEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [m.predict(X) for m in self.models]
        predict = np.mean(predictions, axis=0)
        return predict

    def predict_proba(self, X):
        predictions = [m.predict_proba(X) for m in self.models]
        predict = np.mean(predictions, axis=0)
        return predict


def parse_dt(x):
    if not isinstance(x, str):
        return None
    elif len(x) == len('2010-01-01'):
        return datetime.datetime.strptime(x, '%Y-%m-%d')
    elif len(x) == len('2010-01-01 10:10:10'):
        return datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    else:
        return None


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = df[col_name].apply(lambda x: parse_dt(x))
        df['number_weekday_{}'.format(col_name)] = df[col_name].apply(lambda x: x.weekday())
        df['number_month_{}'.format(col_name)] = df[col_name].apply(lambda x: x.month)
        df['number_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.day)
        df['number_hour_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour)
        df['number_hour_of_week_{}'.format(col_name)] = df[col_name].apply(lambda x: x.hour + x.weekday() * 24)
        df['number_minute_of_day_{}'.format(col_name)] = df[col_name].apply(lambda x: x.minute + x.hour * 60)
    return df
