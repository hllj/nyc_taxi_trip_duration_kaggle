from sklearn.metrics import mean_squared_log_error
import numpy as np

def get_rmsle(Y, Y_pred):
    error = mean_squared_log_error(Y, Y_pred, squared=False)
    return error

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