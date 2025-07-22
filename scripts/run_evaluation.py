# scripts/run_evaluation.py
import yaml
import torch
import joblib
import argparse
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_raw_data
from src.data.features import add_features
from src.data.imager import generate_images_from_df
from src.data.splitter import DataSplitter
from src.data.dataset import ImageDataset
from src.model.architecture import create_cnn_from_config
from src.training.trainer import validate_epoch
from src.optimization.optimizer import HyperparameterOptimizer
from torch.utils.data import DataLoader

def run_evaluation(run_dir: Path):
    """
    Loads the best model from an optimization run and evaluates it on the hold-out test set.
    """
    print("=" * 80)
    print(f"🚀 Starting Final Evaluation for Run: {run_dir.name}")
    print("=" * 80)

    # --- 1. Load Artifacts and Reconstruct BEST Config FIRST ---
    print("\n[1/4] Loading artifacts and reconstructing best configuration...")
    base_config_path = run_dir / "base_config.yaml"
    study_path = run_dir / "optuna_study.pkl"
    
    if not base_config_path.exists() or not study_path.exists():
        print(f"❌ Error: base_config.yaml or optuna_study.pkl not found in {run_dir}")
        return

    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    study = joblib.load(study_path)
    best_trial = study.best_trial
    print(f"✅ Best trial found: #{best_trial.number} with value {best_trial.value:.4f}")
    
    # Lade die für diesen Trial gespeicherte, exakte Konfiguration
    best_trial_dir = run_dir / f"trial_{best_trial.number:03d}"
    best_config_path = best_trial_dir / "config.yaml"
    
    if not best_config_path.exists():
        print(f"❌ Error: The specific config file for the best trial was not found at {best_config_path}")
        # Fallback: Rekonstruieren, falls das Speichern mal fehlgeschlagen ist
        print("Attempting to reconstruct config from trial object...")
        optimizer_for_reconstruction = HyperparameterOptimizer(base_config, mode='model_optimization')
        best_config = optimizer_for_reconstruction.suggest_and_update_config(best_trial)
    else:
        print(f"✅ Loading best model's specific configuration from: {best_config_path}")
        with open(best_config_path, 'r') as f:
            best_config = yaml.safe_load(f)

    model_path = best_trial_dir / "model.pt"
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        return

    # --- 2. Prepare Test Data using the BEST CONFIG ---
    print("\n[2/4] Preparing unseen test data using the best trial's configuration...")
    df_raw, _ = load_raw_data(best_config)
    df_featured = add_features(df_raw.copy(), best_config)
    
    chron_config = best_config['splitting']['chronological']
    test_start_date = pd.to_datetime(chron_config['test_start_date']).tz_localize('UTC')
    
    essential_cols = ['open', 'high', 'low', 'close', 'volume']
    active_feature_keys = best_config.get('features', {}).keys()
    cols_to_keep = essential_cols.copy()
    for col in df_featured.columns:
        prefix = col.split('_')[0].lower()
        if prefix in active_feature_keys:
            cols_to_keep.append(col)

    # Stelle sicher, dass keine Duplikate vorhanden sind und die Reihenfolge erhalten bleibt
    df_filtered = df_featured[list(dict.fromkeys(cols_to_keep))]
    
    lookback_for_test = best_config['imaging']['lookback_period']
    df_for_imaging = df_filtered[df_filtered.index >= (test_start_date - pd.DateOffset(days=lookback_for_test))]
    
    eval_data_dir = run_dir / "final_evaluation_data"
    image_paths, labels, dates = generate_images_from_df(df_for_imaging, best_config, str(eval_data_dir))
    
    if not image_paths:
        print("❌ Error: No images could be generated for the test set.")
        return

    artifacts_df = pd.DataFrame({'date': dates, 'image_path': image_paths, 'label': labels})
    
    data_splitter = DataSplitter(best_config)
    data_splits = data_splitter.split_data(artifacts_df)
    test_data = data_splits.get('test')

    if not test_data or not test_data['paths']:
        print("❌ Error: No test data could be generated. Check your test_start_date in the config.")
        return
    
    print(f"✅ Test set created with {len(test_data['paths'])} samples.")

    # --- 3. Load Model and Create DataLoader ---
    print("\n[3/4] Loading model and creating test DataLoader...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    test_dataset = ImageDataset(test_data['paths'], np.array(test_data['labels']))
    input_shape = test_dataset[0][0].shape
    
    model = create_cnn_from_config(best_config, input_shape)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("✅ Best model's weights loaded successfully.")

    test_loader = DataLoader(test_dataset, batch_size=best_config['training']['batch_size'])

    # --- 4. Run Evaluation ---
    print("\n[4/4] Evaluating model performance on the test set...")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    final_metrics = validate_epoch(model, test_loader, loss_fn, device)

    print("\n" + "="*80)
    print("🏆 FINAL EVALUATION RESULTS 🏆")
    print("="*80)
    print(f"  - Test Set Loss:      {final_metrics['val_loss']:.4f}")
    print(f"  - Test Set Accuracy:  {final_metrics['val_accuracy']:.2f}%")
    print(f"  - Test Set F1-Score:    {final_metrics['f1_score']:.4f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the best model from an Optuna run on the test set.")
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the completed optimization run directory (e.g., runs/model_optimization_...).'
    )
    args = parser.parse_args()
    
    run_evaluation(Path(args.run_dir))