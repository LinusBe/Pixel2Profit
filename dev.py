# src/pipeline.py

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
import shutil
from datetime import datetime
import joblib
import optuna
import traceback
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
warnings.filterwarnings('ignore')

from src.data.loader import load_raw_data
from src.data.features import add_features
from src.data.imager import generate_images_from_df
from src.data.splitter import DataSplitter
from src.data.dataset import ImageDataset
from src.model.architecture import create_cnn_from_config
from src.training.trainer import train_epoch, validate_epoch
from src.optimization.optimizer import HyperparameterOptimizer

class PixelToProfitPipeline:
    """
    Orchestrates the simplified ML workflow for educational purposes.
    """
    def __init__(self, config_path: str):
        print("=" * 80); print("Pixel-to-Profit: Student Edition Pipeline"); print("=" * 80)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"✅ Configuration loaded: {config_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Computing Device: {self.device}")
        
        self.run_dir = None
        self.df_raw = None # KORREKTUR: Wird jetzt einmal initialisiert
        self.optimizer = None
        print("-" * 80)

    def run_optimization(self):
        """
        The main entry point to run the hyperparameter optimization study.
        """
        run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = Path("runs") / f"model_optimization_{run_timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Run directory created: {self.run_dir}")
        shutil.copy(self.config_path, self.run_dir / "base_config.yaml")

        # KORREKTUR: Daten werden jetzt EINMAL vor der Optimierung geladen.
        self._step_data_loading()

        print("\n[2/2] MODEL TRAINING & OPTIMIZATION")
        self.optimizer = HyperparameterOptimizer(self.config, mode='model_optimization')
        
        best_trial = self.optimizer.optimize(self._objective)

        if not best_trial:
            print("❌ Optuna study did not find a best trial. Aborting.")
            return
        
        self._save_optimization_results(best_trial)
        
        print("\n" + "="*80 + "\n✅ PIPELINE SUCCESSFULLY COMPLETED!\n" + "="*80)
        print(f"Find all results in: {self.run_dir}")

    def _objective(self, trial: optuna.Trial, trial_config: Dict) -> float:
        """Defines the logic for a single optimization trial for Optuna."""
        trial_dir = self.run_dir / f"trial_{trial.number:03d}"
        print(f"\n--- Optuna Trial {trial.number} | Params: {trial.params} ---")
        
        df_featured_trial, artifacts_df, data_splits_trial, trial_model = None, None, None, None
        
        try:
            # KORREKTUR: Die Datenvorbereitung nutzt jetzt den vorgeladenen `self.df_raw` DataFrame.
            df_featured_trial, artifacts_df, data_splits_trial = self._prepare_data_for_trial(trial_config, trial_dir)
            if not data_splits_trial:
                raise optuna.exceptions.TrialPruned("Data preparation failed.")

            trial_model, model_metrics = self._execute_training_run(trial_config, data_splits_trial, trial)
            if not trial_model:
                raise optuna.exceptions.TrialPruned("Model training failed.")
            
            torch.save(trial_model.state_dict(), trial_dir / "model.pt")
            
            objective_metric_name = self.optimizer.opt_config['objective_metric']
            metric_value = model_metrics.get(objective_metric_name)

            if metric_value is None:
                raise optuna.exceptions.TrialPruned(f"Metric '{objective_metric_name}' not found.")

            print(f"✅ Trial {trial.number} Result for '{objective_metric_name}': {metric_value:.4f}")
            return metric_value

        except optuna.exceptions.TrialPruned as e:
            print(f"⏩ Trial {trial.number} pruned: {e}")
            raise e
        except Exception as e:
            print(f"❌ Error in Trial {trial.number}: {e}"); traceback.print_exc()
            raise optuna.exceptions.TrialPruned("Trial failed with an unexpected exception.")
        finally:
            print(f"--- Cleaning up memory for trial {trial.number} ---")
            del df_featured_trial, artifacts_df, data_splits_trial, trial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # NEU: Effiziente Datenlade-Methode aus der großen Pipeline
    def _step_data_loading(self):
        """Loads the raw time series data ONCE."""
        print("\n[1/2] DATA LOADING & PREPARATION")
        self.df_raw, _ = load_raw_data(self.config)
        print(f"✅ Raw data loaded: {len(self.df_raw)} samples")

    # (Die anderen Methoden wie _execute_training_run, _plot_training_history, _save_optimization_results etc. bleiben weitgehend gleich)
    # ...
    
    # KORREKTUR: Effizientere DataLoader-Erstellung
    def _prepare_dataloaders(self, config: Dict, data_splits: Dict, split_type: str) -> Optional[DataLoader]:
        """Prepares a PyTorch DataLoader for a specific data split."""
        split_data = data_splits.get(split_type)
        if not split_data or not split_data.get('paths'):
            return None
        
        batch_size = config.get('training', {}).get('batch_size', 32)
        # KORREKTUR: num_workers=0 während der Optimierung kann Konflikte vermeiden,
        # aber wir lassen es auf 2, da die Hauptlast (I/O) jetzt wegfällt.
        # pin_memory=True ist der Schlüssel für schnellen GPU-Transfer.
        num_workers = 2 
        pin_memory = (self.device == "cuda")

        dataset = ImageDataset(split_data['paths'], np.array(split_data['labels']))
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split_type == 'train'), num_workers=num_workers, pin_memory=pin_memory)

    def _get_input_shape_from_loader(self, loader: DataLoader) -> Tuple[int, int, int]:
        """Gets the input tensor shape from a DataLoader."""
        return loader.dataset[0][0].shape

    # (Restliche Methoden unverändert hier einfügen)
    # ...