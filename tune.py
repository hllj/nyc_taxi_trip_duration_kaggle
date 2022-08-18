import json
import logging
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler
from utils.utils_tune import get_param

from utils.utils_model import get_feature, get_target, val

from utils.utils import one_hot_encoding

log = logging.getLogger(__name__)

def objective(trial, X, Y, cfg):
    param = get_param(trial, cfg['tune']['name'])
    
    model = hydra.utils.instantiate({'_target_': cfg['tune']['model']}, **param)
    cv = hydra.utils.instantiate(cfg['cv'])
    
    error = cross_val_score(model, X, Y, cv=cv, scoring=val).mean()
    
    return error

def log_study(output_dir, study):
    trials = study.trials
    with open(os.path.join(output_dir, 'trials.csv'), 'w') as f:
        f.write('id,param,value\n')
        for trial in trials:
            id = trial.number
            param = str(trial.params)
            value = trial.values[0]
            f.write(f'{id},{param},{value}\n')
    best_params = study.best_trial.params
    with open(os.path.join(output_dir, 'best.json'), 'w') as f:
        json.dump(best_params, f, indent = 4)

@hydra.main(version_base=None, config_path="configs", config_name="tune")
def main(cfg: DictConfig) -> None:
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
    
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler()
    )
    study.optimize(
        lambda trial: objective(trial, X_train, Y_train, cfg),
        n_trials=cfg['tune']['n_trials'],
    )
    log.info(f'Number of finished trials: {len(study.trials)}')
    log.info(f'Best trial: {study.best_trial.params}')
    output_dir = HydraConfig.get().runtime.output_dir
    log_study(output_dir, study)


if __name__ == '__main__':
    main()