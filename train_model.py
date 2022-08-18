import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.utils import create_submission, one_hot_encoding
from utils.utils_model import data_normalization, get_feature, get_pred, get_target, train, val

log = logging.getLogger(__name__)

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

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=34)
    
    X_test = get_feature(df_test, feature_columns)
    scaler = hydra.utils.instantiate(cfg['model']['scaler'])
    X_train, X_test, scaler = data_normalization(X_train, X_test, scaler)
    X_val = scaler.transform(X_val)

    model = hydra.utils.instantiate(cfg['model']['model'])

    model, train_rmsle = train(X_train, Y_train, model)
    log.info(f"Train RMSLE : {train_rmsle}")

    val_rmsle = val(model, X_val, Y_val)
    log.info(f"Validation RMSLE : {val_rmsle}")
    
    Y_test_pred = get_pred(X_test, model)
    test_data_path = cfg['process']['raw_test']
    submission_name = f"{cfg['model']['name']}_{type(model).__name__}"
    output_dir = HydraConfig.get().runtime.output_dir
    create_submission(output_dir, Y_test_pred, submission_name, test_data_path)
    log.info(f"Create submission {submission_name}")


if __name__ == "__main__":
    main()