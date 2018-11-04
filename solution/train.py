import argparse
import os
import numpy as np
import pandas as pd
import pickle
import time

# from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import StandardScaler

from utils import transform_datetime_features, ModelsEnsemble

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))
ONEHOT_MAX_UNIQUE_VALUES = 30
TARGET_ENCODING_MAX_VALUES = 50
BIG_DATASET_SIZE = 2000 * 1024 * 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()

    df = pd.read_csv(args.train_csv)
    df_y = df.target

    df_X = df.drop('target', axis=1)
    is_big = df_X.memory_usage().sum() > BIG_DATASET_SIZE

    print('Dataset read, shape {}'.format(df_X.shape))

    # drop constant features
    constant_columns = [
        col_name
        for col_name in df_X.columns
        if df_X[col_name].nunique() == 1
    ]
    df_X.drop(constant_columns, axis=1, inplace=True)

    # dict with data necessary to make predictions
    model_config = {}
    model_config['categorical_values'] = {}
    model_config['is_big'] = is_big

    if is_big:
        # missing values
        if any(df_X.isnull()):
            model_config['missing'] = True
            df_X.fillna(-1, inplace=True)

        new_feature_count = min(df_X.shape[1],
                                int(df_X.shape[1] / (df_X.memory_usage().sum() / BIG_DATASET_SIZE)))
        # take only high correlated features
        correlations = np.abs([
            np.corrcoef(df_y, df_X[col_name])[0, 1]
            for col_name in df_X.columns if col_name.startswith('number')
        ])
        new_columns = df_X.columns[np.argsort(correlations)[-new_feature_count:]]
        df_X = df_X[new_columns]

    else:
        # features from datetime
        df_X = transform_datetime_features(df_X)

        # categorical encoding
        categorical_values = {}
        mean_encoding = {"unknown": df_y.mean()}
        mean_encoding_values = {}

        for col_name in list(df_X.columns):
            col_unique_values = df_X[col_name].unique()

            # Mean target encoding
            if 2 < len(col_unique_values) <= TARGET_ENCODING_MAX_VALUES:
                mean_encoding_values[col_name] = col_unique_values
                df_X["number_%s_mean_encoding" % col_name] = mean_encoding["unknown"]
                for unique_value in col_unique_values:
                    mean_encoding["%s_%s" % (col_name, unique_value)] = df_y[df_X[col_name] == unique_value].mean()
                    df_X["number_%s_mean_encoding" % col_name][df_X[col_name] == unique_value] = \
                        mean_encoding["%s_%s" % (col_name, unique_value)]

            # One hot encoding
            if 2 < len(col_unique_values) <= ONEHOT_MAX_UNIQUE_VALUES:
                categorical_values[col_name] = col_unique_values
                for unique_value in col_unique_values:
                    df_X['onehot_{}={}'.format(col_name, unique_value)] = (df_X[col_name] == unique_value).astype(int)

        model_config['categorical_values'] = categorical_values
        model_config['mean_encoding'] = mean_encoding
        model_config['mean_encoding_values'] = mean_encoding_values

        # missing values
        if any(df_X.isnull()):
            model_config['missing'] = True
            df_X.fillna(-1, inplace=True)

    # use only numeric columns
    used_columns = [
        col_name
        for col_name in df_X.columns
        if col_name.startswith('number') or col_name.startswith('onehot')
    ]
    df_X = df_X[used_columns]

    # Finding most informative features by coefficients of liner regression
    if args.mode == 'regression':
        model = Ridge(alpha=0.3, copy_X=False)
    else:
        model = LogisticRegression(C=0.3, n_jobs=-1)

    model.fit(df_X, df_y)

    # Generate new features from most informative features by pair-wise division
    feature_generation_columns = []
    for r, i in sorted(zip(model.coef_, df_X.columns))[:10]:
        feature_generation_columns.append(i)

    for i in feature_generation_columns:
        for j in feature_generation_columns:
            if i == j:
                continue
            k = df_X[j]
            k[k == 0] = 0.0001

            df_X["number_%s_%s" % (i, j)] = df_X[i] / k

    df_X = df_X.values
    model_config['used_columns'] = used_columns
    model_config['feature_generation_columns'] = feature_generation_columns

    # scaling
    scaler = StandardScaler(copy=False)
    df_X = scaler.fit_transform(df_X)
    model_config['scaler'] = scaler

    # fitting
    model_config['mode'] = args.mode

    kf = KFold(n_splits=3)
    models = []

    if not is_big:
        for i, (train, test) in enumerate(kf.split(df_X)):
            print("FOLD ", i)

            if args.mode == 'regression':
                model = LGBMRegressor(reg_alpha=0.3, reg_lambda=0.1, min_child_weight=10,
                                      zero_as_missing=True, learning_rate=0.01, num_leaves=100,
                                      feature_fraction=0.7, bagging_fraction=0.7, n_estimators=800,
                                      n_jobs=-1, min_child_samples=30)
            else:
                model = LGBMClassifier(reg_alpha=0.3, reg_lambda=0.1, min_child_weight=10,
                                       zero_as_missing=True, learning_rate=0.01, num_leaves=100,
                                       feature_fraction=0.7, bagging_fraction=0.7, n_estimators=800,
                                       n_jobs=-1, min_child_samples=30)

            train_x, test_x, train_y, test_y = df_X[train], df_X[test], df_y[train], df_y[test]
            # train_test_split(df_X, df_y, test_size=0.15)

            model.fit(train_x, train_y, eval_set=(test_x, test_y), early_stopping_rounds=7)
            models.append(model)

        model = ModelsEnsemble(models)
    else:
        if TIME_LIMIT > 5 * 60:
            if args.mode == 'regression':
                model = LGBMRegressor(reg_alpha=0.3, reg_lambda=0.1, min_child_weight=10,
                                      zero_as_missing=True, learning_rate=0.01, num_leaves=200,
                                      feature_fraction=0.7, bagging_fraction=0.7, n_estimators=800,
                                      n_jobs=-1, min_child_samples=60)
            else:
                model = LGBMClassifier(reg_alpha=0.3, reg_lambda=0.1, min_child_weight=10,
                                       zero_as_missing=True, learning_rate=0.01, num_leaves=200,
                                       feature_fraction=0.7, bagging_fraction=0.7, n_estimators=800,
                                       n_jobs=-1, min_child_samples=60)

            train_x, test_x, train_y, test_y = train_test_split(df_X, df_y, test_size=0.15)
            model.fit(train_x, train_y, eval_set=(test_x, test_y), early_stopping_rounds=7)
        else:
            if args.mode == 'regression':
                model = Ridge(alpha=0.2, copy_X=False)
            else:
                model = LogisticRegression(C=0.2, n_jobs=-1)

            model.fit(df_X, df_y)

    model_config['model'] = model

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {}'.format(time.time() - start_time))
