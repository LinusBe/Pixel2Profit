# src/optimization/optimizer.py
import optuna
import copy
from typing import Dict, Any, Callable

class HyperparameterOptimizer:
    """
    Manages the hyperparameter optimization process using Optuna.
    This simplified version is focused on model and training parameters.
    """
    def __init__(self, config: Dict[str, Any], mode: str):
        """Initializes the HyperparameterOptimizer."""
        self.base_config = config
        self.opt_config = config['optimization'][mode]
        self.search_space_config = config['optimization'].get('search_space', {})
        self.study = self._create_study()

    def _create_study(self) -> optuna.Study:
        """Creates a new Optuna study based on the objective metric."""
        metric = self.opt_config['objective_metric']
        direction = 'minimize' if 'loss' in metric else 'maximize'
        print(f"Creating Optuna study to '{direction}' the metric '{metric}'.")
        return optuna.create_study(direction=direction, pruner=optuna.pruners.MedianPruner())

    def suggest_and_update_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggests parameters for a trial and updates the configuration."""
        params = {}
        for name, conf in self.search_space_config.items():
            if conf['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, conf['choices'])
            elif conf['type'] == 'int':
                params[name] = trial.suggest_int(name, conf['low'], conf['high'], step=conf.get('step', 1))
            elif conf['type'] == 'float':
                params[name] = trial.suggest_float(name, conf['low'], conf['high'], log=conf.get('log', False))

        if params.get('use_macd') and params.get('macd_fast') >= self.base_config['features']['macd'][0]['slow']:
            raise optuna.exceptions.TrialPruned("Pruning trial due to invalid MACD params (fast >= slow).")

        trial_config = copy.deepcopy(self.base_config)
        
        # Update features based on 'use_...' params
        active_features = {'ohlc': [{}], 'volume': [{}]}
        for feature_key in ['rsi', 'stoch', 'ao', 'sma', 'ema', 'macd', 'vortex', 'bbands', 'atr', 'obv']:
            if params.get(f'use_{feature_key}', False):
                active_features[feature_key] = self.base_config['features'][feature_key]
        trial_config['features'] = active_features

        if params.get('use_rsi'):
            trial_config['features']['rsi'][0]['length'] = params.get('rsi_length', 14)

        # Use the new, robust helper function for all updates
        self._update_nested_value(trial_config, ['imaging', 'lookback_period'], params.get('lookback_period'))
        self._update_nested_value(trial_config, ['labeling', 'horizon'], params.get('horizon'))
        self._update_nested_value(trial_config, ['training', 'batch_size'], params.get('batch_size'))
        self._update_nested_value(trial_config, ['training', 'optimizer', 'type'], params.get('optimizer_type'))
        self._update_nested_value(trial_config, ['training', 'optimizer', 'params', 'lr'], params.get('lr'))
        
        # --- KORREKTUR: Dynamisches Finden der Layer-Indizes ---
        dense_idx, dropout_idx = -1, -1
        # Finde den ersten Dense-Layer (nicht den Output-Layer) und den Dropout-Layer
        for i, layer_conf in enumerate(trial_config['model']['architecture']):
            if layer_conf['layer'] == 'Dense' and dense_idx == -1:
                dense_idx = i
            if layer_conf['layer'] == 'Dropout':
                dropout_idx = i
        
        # Update architecture params only if the layers were found
        self._update_nested_value(trial_config, ['model', 'architecture', 0, 'params', 'filters'], params.get('filters_block1'))
        self._update_nested_value(trial_config, ['model', 'architecture', 0, 'params', 'kernel_size'], [params.get('kernel_size_block1')]*2 if params.get('kernel_size_block1') else None)
        self._update_nested_value(trial_config, ['model', 'architecture', 4, 'params', 'filters'], params.get('filters_block2'))
        self._update_nested_value(trial_config, ['model', 'architecture', 4, 'params', 'kernel_size'], [params.get('kernel_size_block2')]*2 if params.get('kernel_size_block2') else None)
        if dense_idx != -1:
            self._update_nested_value(trial_config, ['model', 'architecture', dense_idx, 'params', 'units'], params.get('dense_units'))
        if dropout_idx != -1:
            self._update_nested_value(trial_config, ['model', 'architecture', dropout_idx, 'params', 'rate'], params.get('dropout_rate'))

        return trial_config

    def _update_nested_value(self, config_dict: Dict, path: list, value: Any):
        """
        Helper to robustly update a value in a nested structure of dicts and lists.
        """
        if value is None:
            return
        
        temp_dict = config_dict
        for key in path[:-1]:
            temp_dict = temp_dict[key]
        
        temp_dict[path[-1]] = value

    def optimize(self, objective_function: Callable[[optuna.Trial, Dict], float]) -> optuna.Trial:
        """Starts the optimization process."""
        def objective_wrapper(trial: optuna.Trial) -> float:
            trial_config = self.suggest_and_update_config(trial)
            return objective_function(trial, trial_config)

        self.study.optimize(objective_wrapper, n_trials=self.base_config['optimization'].get('n_trials', 50))
        
        print("\n--- Optimization Finished ---")
        return self.study.best_trial