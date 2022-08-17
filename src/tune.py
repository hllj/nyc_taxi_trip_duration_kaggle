import logging
import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import optuna

from utils.utils_model import get_feature, get_target, val

from utils.utils import one_hot_encoding

log = logging.getLogger(__name__)

def objective(trial, X, Y, cfg):
    param = {
        'tree_method':'gpu_hist',  # Use GPU acceleration
        'lambda': trial.suggest_loguniform(
            'lambda', 1e-3, 10.0
        ),
        'alpha': trial.suggest_loguniform(
            'alpha', 1e-3, 10.0
        ),
        'colsample_bytree': trial.suggest_categorical(
            'colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1.0]
        ),
        'subsample': trial.suggest_categorical(
            'subsample', [0.6,0.7,0.8,1.0]
        ),
        'learning_rate': trial.suggest_categorical(
            'learning_rate', [0.008,0.009,0.01,0.012,0.014,0.016,0.018, 0.02]
        ),
        'n_estimators': trial.suggest_categorical(
            "n_estimators", [150, 200, 300, 3000]
        ),
        'max_depth': trial.suggest_categorical(
            'max_depth', [4,5,7,9,11,13,15,17]
        ),
        'random_state': 42,
        'min_child_weight': trial.suggest_int(
            'min_child_weight', 1, 300
        ),
    }
    model = hydra.utils.instantiate({'_target_': cfg['tune']['model']}, **param)
    cv = hydra.utils.instantiate(cfg['model_select']['cv'])
    
    error = cross_val_score(model, X, Y, cv=cv, scoring=val).mean()
    
    return error

@hydra.main(version_base=None, config_path="configs", config_name="main")
def tune(cfg: DictConfig) -> None:
    target_column = cfg['tune']['target_column']
    categorical_column = cfg['tune']['categorical_column']
    numerical_column = cfg['tune']['numerical_column']
    feature_columns = categorical_column + numerical_column
    df_train = pd.read_csv(cfg['process']['output_process_train'])

    if cfg['tune']['encoding']:
        df_train = one_hot_encoding(df_train, categorical_column)
    
    X_train = get_feature(df_train, feature_columns)
    Y_train = get_target(df_train, target_column)

    scaler = hydra.utils.instantiate(cfg['tune']['scaler'])

    X_train = scaler.fit_transform(X_train)
    
    cv = hydra.utils.instantiate(cfg['tune']['cv'])
    
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, Y_train, cfg),
        n_trials=cfg['tune']['n_trials'],
        n_jobs=-1
    )
    log.info(f'Number of finished trials: {len(study.trials)}')
    log.info(f'Best trial: {study.best_trial.params}')


if __name__ == '__main__':
    tune()