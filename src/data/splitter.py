# src/data/splitter.py
import pandas as pd
import numpy as np
from typing import Dict, Any

class DataSplitter:
    """
    Performs a chronological data split based on dates in the configuration.
    """
    def __init__(self, config: Dict[str, Any]):
        """Initializes the DataSplitter."""
        self.config = config
        self.split_config = config['splitting']
        self.method = self.split_config['method']

    def split_data(self, artifacts_df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Splits the data into training and validation sets based on chronological cutoff dates.
        """
        print(f"\n--- Splitting data using method: '{self.method}' ---")
        if self.method != "chronological":
            raise ValueError(f"Only 'chronological' splitting method is supported in this version.")

        artifacts_df['date'] = pd.to_datetime(artifacts_df['date'])
        artifacts_df = artifacts_df.set_index('date').sort_index()

        chron_config = self.split_config['chronological']
        train_end = pd.to_datetime(chron_config['train_end_date']).tz_localize('UTC')
        val_end = pd.to_datetime(chron_config['val_end_date']).tz_localize('UTC')
        # NEU: Lese das Startdatum für das Test-Set
        test_start = pd.to_datetime(chron_config['test_start_date']).tz_localize('UTC')

        if not train_end < val_end < test_start:
            raise ValueError("Dates must be in ascending order: train_end_date < val_end_date < test_start_date.")

        train_mask = artifacts_df.index <= train_end
        val_mask = (artifacts_df.index > train_end) & (artifacts_df.index <= val_end)
        # NEU: Erstelle eine Maske für die Testdaten
        test_mask = artifacts_df.index >= test_start
        
        splits = self._create_split_dict_from_masks(artifacts_df, train_mask, val_mask, test_mask)
        print(f"Chronological Split: {len(splits['train']['paths'])} Train, {len(splits['val']['paths'])} Val, {len(splits['test']['paths'])} Test.")
        return splits

    def _create_split_dict_from_masks(self, df: pd.DataFrame, train_mask, val_mask, test_mask) -> Dict[str, Dict[str, np.ndarray]]:
        """Helper function to create the split dictionary from boolean masks."""
        df_reset = df.reset_index()
        return {
            'train': {
                'paths': df_reset[train_mask]['image_path'].tolist(),
                'labels': df_reset[train_mask]['label'].to_numpy()
            },
            'val': {
                'paths': df_reset[val_mask]['image_path'].tolist(),
                'labels': df_reset[val_mask]['label'].to_numpy()
            },
            # NEU: Füge das Test-Set zum Output hinzu
            'test': {
                'paths': df_reset[test_mask]['image_path'].tolist(),
                'labels': df_reset[test_mask]['label'].to_numpy()
            }
        }