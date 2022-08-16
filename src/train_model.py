import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.utils import create_submission, get_rmsle, one_hot_encoding

log = logging.getLogger(__name__)

def data_normalization(X_train, X_test, scaler):
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    return X_train, X_test, scaler

def train(X_train, Y_train, model):
    model.fit(X_train, Y_train)
    train_error = get_rmsle(Y_train, model.predict(X_train))
    return model, train_error

def val(model, X, Y):
    Y_pred = model.predict(X)
    error = get_rmsle(Y, Y_pred)
    return error

def get_target(df, target_column):
    Y = df[target_column].values
    return Y

def get_feature(df, columns):
    X = df[columns].values
    return X

def get_pred(X_test, model):
    Y_test_pred = model.predict(X_test)
    # get inverse log with exp
    Y_test_pred = np.exp(Y_test_pred)
    return Y_test_pred

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg : DictConfig) -> None:
    target_column = cfg['model']['target_column']
    categorical_column = cfg['model']['categorical_column']
    numerical_column = cfg['model']['numerical_column']
    feature_columns = categorical_column + numerical_column
    df_train = pd.read_csv(cfg['process']['output_process_train'])
    df_test = pd.read_csv(cfg['process']['output_process_test'])

    if cfg['model']['encoding']:
        df_train = one_hot_encoding(df_train, categorical_column)
        df_test = one_hot_encoding(df_test, categorical_column)

    X_train = get_feature(df_train, feature_columns)
    Y_train = get_target(df_train, target_column)
    
    X_test = get_feature(df_test, feature_columns)
    scaler = hydra.utils.instantiate(cfg['model']['scaler'])
    X_train, X_test, scaler = data_normalization(X_train, X_test, scaler)

    model = hydra.utils.instantiate(cfg['model']['model'])

    model, train_rmsle = train(X_train, Y_train, model)
    log.info(f"Train RMSLE : {train_rmsle}")
    
    Y_test_pred = get_pred(X_test, model)
    test_data_path = cfg['process']['raw_test']
    submisson_folder = cfg['submission_folder']
    submission_name = f"{cfg['model']['name']}_{type(model).__name__}"
    create_submission(Y_test_pred, submission_name, test_data_path, submisson_folder)
    log.info(f"Create submission {submission_name}")


if __name__ == "__main__":
    main()