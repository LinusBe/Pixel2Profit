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

    This class manages the process from data loading to hyperparameter
    optimization. Its sole purpose is to run an Optuna study to find the
    best model based on validation metrics.
    """
    def __init__(self, config_path: str):
        """Initializes the pipeline by loading the configuration."""
        print("=" * 80); print("Pixel-to-Profit: Student Edition Pipeline"); print("=" * 80)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        print(f"✅ Configuration loaded: {config_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Computing Device: {self.device}")
        
        self.run_dir = None
        self.df_raw = None
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
        """
        Defines the logic for a single optimization trial for Optuna.
        """
        trial_dir = self.run_dir / f"trial_{trial.number:03d}"
        print(f"\n--- Optuna Trial {trial.number} | Params: {trial.params} ---")
        
        # Define variables here to ensure they exist in the finally block's scope
        df_featured_trial, artifacts_df, data_splits_trial, trial_model = None, None, None, None
        
        try:
            # --- Data Preparation for Trial ---
            df_featured_trial, artifacts_df, data_splits_trial = self._prepare_data_for_trial(trial_config, trial_dir)
            if not data_splits_trial:
                raise optuna.exceptions.TrialPruned("Data preparation failed.")

            # --- Model Training for Trial ---
            trial_model, model_metrics = self._execute_training_run(trial_config, data_splits_trial, trial)
            if not trial_model:
                raise optuna.exceptions.TrialPruned("Model training failed.")
            
            # --- Save Artifacts and Get Metric ---
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
            # --- Explicit Memory Cleanup ---
            print(f"--- Cleaning up memory for trial {trial.number} ---")
            del df_featured_trial
            del artifacts_df
            del data_splits_trial
            del trial_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _step_data_loading(self):
        """Loads the raw time series data."""
        print("\n[1/2] DATA LOADING & PREPARATION")
        self.df_raw, _ = load_raw_data(self.config)
        print(f"✅ Raw data loaded: {len(self.df_raw)} samples")

    def _prepare_data_for_trial(self, config: Dict, trial_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[Any]]:
        """Prepares all data artifacts (features, images, splits) for a trial."""
        try:
            trial_dir.mkdir(parents=True, exist_ok=True)
            df_featured = add_features(self.df_raw.copy(), config)
            
            # Keep only columns that are actually used based on the config
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            active_feature_keys = config.get('features', {}).keys()
            cols_to_keep = essential_cols.copy()
            for col in df_featured.columns:
                prefix = col.split('_')[0].lower()
                if prefix in active_feature_keys:
                    cols_to_keep.append(col)
            
            df_for_imaging = df_featured[list(dict.fromkeys(cols_to_keep))]

            images_dir = trial_dir / "images"
            image_paths, labels, dates = generate_images_from_df(df_for_imaging, config, output_dir=str(images_dir))
            if not image_paths: return None, None, None
            
            artifacts_df = pd.DataFrame({'date': dates, 'image_path': image_paths, 'label': labels})
            artifacts_df.to_csv(trial_dir / "artifacts.csv", index=False)

            data_splitter = DataSplitter(config)
            data_splits = data_splitter.split_data(artifacts_df)

            return df_featured, artifacts_df, data_splits
        except Exception as e:
            print(f"❌ Error during data preparation for {trial_dir.name}: {e}"); traceback.print_exc()
            return None, None, None

    def _execute_training_run(self, config: Dict, data_splits: Dict, trial: Optional[optuna.Trial] = None) -> Tuple[Optional[nn.Module], Dict]:
        """Executes the core training and validation loop for a given configuration."""
        train_loader = self._prepare_dataloaders(config, data_splits, 'train')
        val_loader = self._prepare_dataloaders(config, data_splits, 'val')
        if not train_loader or not val_loader:
            print("❗ Train or validation set is empty. Cannot execute training.")
            return None, {}
        
        input_shape = self._get_input_shape_from_loader(train_loader)
        model = create_cnn_from_config(config, input_shape).to(self.device)
        
        train_config = config['training']
        optimizer_params = train_config['optimizer'].get('params', {})
        optimizer = getattr(torch.optim, train_config['optimizer']['type'])(model.parameters(), **optimizer_params)
        
        pos_labels = np.sum(data_splits['train']['labels'])
        neg_labels = len(data_splits['train']['labels']) - pos_labels
        pos_weight = neg_labels / (pos_labels + 1e-8)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        
        es_config = train_config.get('early_stopping', {})
        best_metric_val = -np.inf if es_config.get('monitor') != 'val_loss' else np.inf
        patience_counter = 0
        final_metrics = {}
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'f1_score': []}

        for epoch in range(train_config['epochs']):
            print(f"\n--- Epoch {epoch + 1}/{train_config['epochs']} ---")
            
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, self.device)
            val_metrics = validate_epoch(model, val_loader, loss_fn, self.device)
            
            history['train_loss'].append(train_loss)
            for key in ['val_loss', 'val_accuracy', 'f1_score']:
                history[key].append(val_metrics[key])

            final_metrics = val_metrics

            if es_config.get('enabled'):
                monitor_val = val_metrics.get(es_config['monitor'])
                if trial:
                    trial.report(monitor_val, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                improved = (monitor_val < best_metric_val - es_config.get('min_delta', 0)) if 'loss' in es_config['monitor'] else (monitor_val > best_metric_val + es_config.get('min_delta', 0))
                if improved:
                    best_metric_val, patience_counter = monitor_val, 0
                    print(f"✅ EarlyStopping: New best for '{es_config['monitor']}': {best_metric_val:.4f}")
                else:
                    patience_counter += 1
                    print(f"❗ EarlyStopping: No improvement for {patience_counter} epoch(s). Patience is {es_config['patience']}.")

                if patience_counter >= es_config['patience']:
                    print(f"❗ Early stopping triggered after {epoch + 1} epochs."); break
        
        output_dir = self.run_dir / f"trial_{trial.number:03d}" if trial else self.run_dir
        self._plot_training_history(history, output_dir)

        return model, final_metrics

    def _prepare_dataloaders(self, config: Dict, data_splits: Dict, split_type: str) -> Optional[DataLoader]:
        """Prepares a PyTorch DataLoader for a specific data split."""
        split_data = data_splits.get(split_type)
        if not split_data or not split_data.get('paths'):
            return None
        
        batch_size = config.get('training', {}).get('batch_size', 32)
        dataset = ImageDataset(split_data['paths'], np.array(split_data['labels']))
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split_type == 'train'), num_workers=2, pin_memory=True)

    def _get_input_shape_from_loader(self, loader: DataLoader) -> Tuple[int, int, int]:
        """Gets the input tensor shape from a DataLoader."""
        return loader.dataset[0][0].shape

    def _plot_training_history(self, history: Dict, output_dir: Path):
        """Creates and saves a plot of training and validation performance."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            ax1.plot(history['train_loss'], label='Training Loss', color='blue')
            ax1.plot(history['val_loss'], label='Validation Loss', color='orange')
            ax1.set_ylabel("Loss")
            ax1.set_title("Training & Validation Loss")
            ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)

            ax2.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(history['f1_score'], label='Validation F1-Score', color='red')
            ax2.set_xlabel("Epochs"); ax2.set_ylabel("Accuracy (%)"); ax2_twin.set_ylabel("F1-Score")
            
            lines, labels = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2_twin.legend(lines + lines2, labels + labels2, loc=0); ax2.grid(True, linestyle='--', alpha=0.6)
            
            plt.tight_layout()
            plot_path = output_dir / "training_performance.png"
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"✅ Training performance plot saved to: {plot_path}")
        except Exception as e:
            print(f"❗ Could not create training performance plot: {e}")

    def _save_optimization_results(self, best_trial: optuna.Trial):
        """Saves the Optuna study and visualization plots."""
        joblib.dump(self.optimizer.study, self.run_dir / "optuna_study.pkl")
        try:
            fig = optuna.visualization.plot_optimization_history(self.optimizer.study)
            fig.write_image(self.run_dir / "optuna_history.png")
            
            fig = optuna.visualization.plot_slice(self.optimizer.study)
            fig.write_image(self.run_dir / "optuna_slice.png")

            fig = optuna.visualization.plot_param_importances(self.optimizer.study)
            fig.write_image(self.run_dir / "optuna_param_importances.png")
            
            print(f"✅ Optuna analysis plots saved in: {self.run_dir}")
        except Exception as e:
            print(f"❗ Could not create Optuna plots: {e}")