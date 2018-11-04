import argparse
import os
import numpy as np
import pandas as pd
import pickle
import time

from utils import transform_datetime_features

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    df = pd.read_csv(args.test_csv)
    print('Dataset read, shape {}'.format(df.shape))

    if not model_config['is_big']:
        # features from datetime
        df = transform_datetime_features(df)

        # categorical encoding
        for col_name, unique_values in model_config['mean_encoding_values'].items():
            df["number_%s_mean_encoding" % col_name] = model_config["mean_encoding"]["unknown"]

            for unique_value in unique_values:
                k = "%s_%s" % (col_name, unique_value)
                if k in model_config["mean_encoding"]:
                    df["number_%s_mean_encoding" % col_name][df[col_name] == unique_value] =\
                        model_config["mean_encoding"][k]

        for col_name, unique_values in model_config['categorical_values'].items():
            for unique_value in unique_values:
                df['onehot_{}={}'.format(col_name, unique_value)] = (df[col_name] == unique_value).astype(int)

    # missing values
    if model_config['missing']:
        df.fillna(-1, inplace=True)
    elif any(df.isnull()):
        df.fillna(value=df.mean(axis=0), inplace=True)

    # filter columns
    used_columns = model_config['used_columns']

    line_id = df['line_id']
    df = df[used_columns]
    for i in model_config['feature_generation_columns']:
        for j in model_config['feature_generation_columns']:
            if i == j:
                continue
            k = df[j]
            k[k == 0] = 0.0001

            df["number_%s_%s" % (i, j)] = df[i] / k

    # scale
    X_scaled = model_config['scaler'].transform(df)

    model = model_config['model']
    if model_config['mode'] == 'regression':
        df['prediction'] = model.predict(X_scaled)
    elif model_config['mode'] == 'classification':
        df['prediction'] = model.predict_proba(X_scaled)[:, 1]

    df['line_id'] = line_id
    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)

    print('Prediction time: {}'.format(time.time() - start_time))
