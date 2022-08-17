import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import cross_val_score

from utils.utils_model import get_feature, get_target, val

from utils.utils import one_hot_encoding

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="main")
def model_select(cfg: DictConfig) -> None:
    target_column = cfg['model_select']['target_column']
    categorical_column = cfg['model_select']['categorical_column']
    numerical_column = cfg['model_select']['numerical_column']
    feature_columns = categorical_column + numerical_column
    df_train = pd.read_csv(cfg['process']['output_process_train'])

    if cfg['model_select']['encoding']:
        df_train = one_hot_encoding(df_train, categorical_column)
    
    X_train = get_feature(df_train, feature_columns)
    Y_train = get_target(df_train, target_column)

    scaler = hydra.utils.instantiate(cfg['model_select']['scaler'])

    X_train = scaler.fit_transform(X_train)
    
    Models = []
    Errors = []
    cv = hydra.utils.instantiate(cfg['model_select']['cv'])

    names = cfg['model_select']['names']
    models = cfg['model_select']['models']
    print(names, models)
    for name, module in zip(names, models):
        model = hydra.utils.instantiate({'_target_': module})
        error = cross_val_score(model, X_train, Y_train, cv=cv, scoring=val).mean()
        log.info(f"Model {name}, error : {error}")
        Models.append(name)
        Errors.append(error)

    evaluation = pd.DataFrame({'Model': Models,
                           'Error': Errors})
    name = cfg['model_select']['output']
    evaluation.to_csv(name)

if __name__ == '__main__':
    model_select()