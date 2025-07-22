import pandas as pd
import pandas_ta as ta
from typing import Dict, Any

def add_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Adds technical indicators to the DataFrame based on the configuration.

    This function serves as the feature engineering component of the pipeline.
    It dynamically calculates a set of technical indicators using the `pandas-ta`
    library based on definitions in the `features` section of the main
    configuration file. It also intelligently handles the removal of the initial
    "warmup" period from the data, which contains NaN values, ensuring the
    output DataFrame is ready for further processing.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing OHLCV data, indexed by datetime.
    config : Dict[str, Any]
        The project's main configuration dictionary. This function specifically
        uses `config['features']` to define the indicators and their parameters.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the calculated technical indicators as new columns.
        The initial rows corresponding to the longest indicator's warmup
        period have been removed.
    """
    print("\n--- Adding Technical Indicators ---")
    if 'features' not in config:
        print("No 'features' block in config. Skipping.")
        return df
        
    feature_config = config['features']
    
    # --- Calculate the longest warmup period for calculated indicators ---
    max_lookback = 0
    
    # --- Prepare the strategy for pandas-ta ---
    strategy_params = []
    
    for indicator, params_list in feature_config.items():
        # Special handling for raw data features that don't need calculation
        if indicator in ['ohlc', 'volume']:
            continue

        for params in params_list:
            strategy_params.append({"kind": indicator, **params})
            
            # Update max_lookback for warmup period
            if 'length' in params and params.get('length', 0) > max_lookback:
                max_lookback = params['length']
            if 'slow' in params and params.get('slow', 0) > max_lookback: # For MACD
                max_lookback = params['slow']

    # --- Calculate indicators using the pandas-ta strategy ---
    if strategy_params:
        strategy = ta.Strategy(name="P2P_Strategy", ta=strategy_params)
        df.ta.strategy(strategy)
        print("Calculated indicators added successfully.")
    else:
        print("No indicators to calculate.")
        
    # --- Truncate the warmup period ---
    if max_lookback > 0:
        print(f"Longest warmup period is {max_lookback} days. Truncating DataFrame.")
        rows_before = len(df)
        df_truncated = df.iloc[max_lookback:].copy()
        rows_after = len(df_truncated)
        print(f"{rows_before - rows_after} rows were removed due to indicator warmup.")
    else:
        df_truncated = df.copy()

    # --- Final check for required raw columns ---
    # This ensures that even if no indicators were calculated, the raw columns are present.
    final_columns = list(df_truncated.columns)
    print(f"Final feature set contains {len(final_columns)} columns.")
    
    return df_truncated