# =============================================================================
# FILE: /src/data/loader.py
# =============================================================================
import pandas as pd
from typing import Tuple, Dict, Any

def load_raw_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Loads and validates the raw financial time-series data from a CSV file.

    This function reads data specifications from the configuration dictionary,
    such as the file path and required columns. It performs several crucial
    preprocessing steps: it validates column existence, standardizes column
    names, converts the date column to timezone-aware Timestamps (UTC), and
    sets it as the DataFrame's index. Crucially, it sorts the DataFrame by
    this index to ensure correct chronological order for time-series analysis.

    Parameters
    ----------
    config : Dict[str, Any]
        The project's main configuration dictionary. It must contain the `data`
        key with sub-keys `raw_csv_path`, `required_columns`, and
        `datetime_column`.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        A tuple containing:
        - The loaded, sorted, and indexed DataFrame.
        - A dictionary analyzing the percentage of NaN values for any
          column that contains them.

    Raises
    ------
    ValueError
        If any of the columns specified in `config['data']['required_columns']`
        are not found in the CSV file.
    FileNotFoundError
        If the path specified in `config['data']['raw_csv_path']` does not exist.
    """
    data_config = config['data']
    raw_path = data_config['raw_csv_path']
    required_cols = data_config['required_columns']
    dt_col = data_config['datetime_column']

    print(f"\nLade Rohdaten von: {raw_path}")
    df = pd.read_csv(raw_path)

    df.columns = df.columns.str.strip()

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Folgende Spalten fehlen in der CSV-Datei: {missing_cols}")

    df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
    df = df.set_index(dt_col)

    # Sort the DataFrame by its datetime index to ensure chronological order.
    df = df.sort_index()

    nan_analysis = {col: df[col].isna().mean() * 100 for col in df.columns if df[col].isna().any()}

    return df, nan_analysis