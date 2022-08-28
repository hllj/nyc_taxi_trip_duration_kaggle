
def get_param(trial, name):
    if name == 'baseline':
        param = {}
    if name == 'xgboost':
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
    return param
