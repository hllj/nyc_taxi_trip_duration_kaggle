from distutils.command.clean import clean
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

from utils import haversine

def clean_data(df_train, df_test):
    df_train.drop('id',inplace=True,axis=1)
    df_train.drop('dropoff_datetime',inplace=True,axis=1)
    df_test.drop('id',inplace=True,axis=1)
    return df_train, df_test
    
def transform(df_train, df_test):
    df_train['trip_duration'] = np.log(df_train['trip_duration'])
    return df_train, df_test

def feature_engineer(df_train, df_test):
    # Convert
    df_train["pickup_datetime"] = pd.to_datetime(df_train["pickup_datetime"])
    df_test["pickup_datetime"] = pd.to_datetime(df_test["pickup_datetime"])
    
    # Feature Engineering
    df_train['day_of_week'] = df_train['pickup_datetime'].dt.day_name()
    df_train['hour_of_day'] = df_train['pickup_datetime'].dt.hour
    df_train['month'] = df_train['pickup_datetime'].dt.month
    df_train['store_and_fwd_flag'] = df_train['store_and_fwd_flag'].map({'N':0,'Y':1})
    df_train['distance'] = df_train.apply(lambda row:haversine(row['pickup_latitude'],row['pickup_longitude'],row['dropoff_latitude'],row['dropoff_longitude']),axis=1)
    
    df_test['day_of_week'] = df_test['pickup_datetime'].dt.day_name()
    df_test['hour_of_day'] = df_test['pickup_datetime'].dt.hour
    df_test['month'] = df_test['pickup_datetime'].dt.month
    df_test['store_and_fwd_flag'] = df_test['store_and_fwd_flag'].map({'N':0,'Y':1})
    df_test['distance'] = df_test.apply(lambda row:haversine(row['pickup_latitude'],row['pickup_longitude'],row['dropoff_latitude'],row['dropoff_longitude']),axis=1)
    
    df_train['day_of_week'] = df_train['day_of_week'].map({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7})
    df_test['day_of_week'] = df_test['day_of_week'].map({'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7})
    
    df_train.drop('pickup_datetime', inplace=True, axis=1)
    df_train.drop('pickup_latitude', inplace=True, axis=1)
    df_train.drop('pickup_longitude', inplace=True, axis=1)
    df_train.drop('dropoff_longitude', inplace=True, axis=1)
    df_train.drop('dropoff_latitude', inplace=True, axis=1)
    
    df_test.drop('pickup_datetime', inplace=True, axis=1)
    df_test.drop('pickup_latitude', inplace=True, axis=1)
    df_test.drop('pickup_longitude', inplace=True, axis=1)
    df_test.drop('dropoff_longitude', inplace=True, axis=1)
    df_test.drop('dropoff_latitude', inplace=True, axis=1)
    
    return df_train, df_test

def read_raw_data(cfg):
    raw_train_path = cfg['process']['raw_train']
    raw_test_path = cfg['process']['raw_test']
    df_train = pd.read_csv(raw_train_path)
    df_test = pd.read_csv(raw_test_path)
    return df_train, df_test

def save_process_data(cfg, df_train, df_test):
    df_train.to_csv(cfg['process']['output_process_train'], index=False)
    df_test.to_csv(cfg['process']['output_process_test'], index=False) 

@hydra.main(version_base=None, config_path="configs", config_name="main")
def process_data(cfg : DictConfig) -> None:
    df_train, df_test = read_raw_data(cfg)
    df_train, df_test = clean_data(df_train, df_test)
    df_train, df_test = transform(df_train, df_test)
    df_train, df_test = feature_engineer(df_train, df_test)
    save_process_data(cfg, df_train, df_test)

if __name__ == "__main__":
    process_data()