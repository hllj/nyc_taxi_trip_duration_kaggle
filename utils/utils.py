import math
import pandas as pd
import os



def haversine(lat1, lon1, lat2, lon2):
     
    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0
 
    # convert to radians
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0
 
    # apply formulae
    a = (pow(math.sin(dLat / 2), 2) +
         pow(math.sin(dLon / 2), 2) *
             math.cos(lat1) * math.cos(lat2))
    rad = 6371
    c = 2 * math.asin(math.sqrt(a))
    return rad * c

def one_hot_encoding(df, categorical_column):
    for column in categorical_column:
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df.drop(column, axis=1)
    return df

def create_submission(output_dir, Y_test_pred, submission_name, test_data_path):
    submission_df = pd.read_csv(test_data_path)
    submission_df['trip_duration'] = Y_test_pred.tolist()
    submission_df = submission_df[['id', 'trip_duration']]
    os.makedirs(os.path.join(output_dir, 'submisson'))
    submission_df.to_csv(os.path.join(output_dir, 'submisson', submission_name + '.csv'), index=False)