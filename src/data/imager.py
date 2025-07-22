import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.gridspec as gridspec

# Define the prefix map once at the module level for reuse
PREFIX_MAP = {
    'BBL': 'bbands', 'BBM': 'bbands', 'BBU': 'bbands', 'BBB': 'bbands', 'BBP': 'bbands',
    'MACD': 'macd', 'MACDH': 'macd', 'MACDS': 'macd',
    'STOCHK': 'stoch', 'STOCHD': 'stoch',
    'VTXP': 'vortex', 'VTXM': 'vortex',
    'AO': 'ao', 'RSI': 'rsi', 'SMA': 'sma', 'EMA': 'ema', 'OBV': 'obv',
    'ATRR': 'atr'
}

def _create_and_save_image(args: Tuple) -> Optional[Tuple[str, int]]:
    """Creates and saves a single financial chart image from a window of data.

    This function is designed to be executed in parallel by a ProcessPoolExecutor.
    It takes a tuple of arguments containing all necessary data and configuration
    for a single image. The process involves normalizing indicators to the price
    scale, plotting all features using matplotlib on a black background,
    converting the plot canvas to a NumPy array, and saving it as a
    compressed .npz file.

    Parameters
    ----------
    args : Tuple
        A tuple containing the necessary arguments:
        (i, window_df_orig, label, config, output_dir).
        - i (int): The index of the image, used for the filename.
        - window_df_orig (pd.DataFrame): The slice of the DataFrame representing
          the lookback period.
        - label (int): The corresponding label for this data window.
        - config (Dict[str, Any]): The main configuration dictionary.
        - output_dir (str): The directory where the .npz file will be saved.

    Returns
    -------
    Optional[Tuple[str, int]]
        A tuple containing the path to the saved .npz file and its
        corresponding label. Returns None if the image could not be created.
    """
    i, window_df_orig, label, config, output_dir = args
    
    # Work on a copy to avoid modifying the original data slice
    window_df = window_df_orig.copy()

    img_config = config['imaging']
    lookback = img_config['lookback_period']
    height = img_config['image_height']
    width = lookback * img_config['pixels_per_day']
    
    channel_mapping = img_config.get('channel_mapping', {})

    def get_color_for_feature(feature_name: str) -> List[float]:
        """Intelligently finds the correct color for a feature from the config.

        It first tries a direct match with the feature name. If not found, it
        uses the PREFIX_MAP to find the base indicator type (e.g., 'SMA_20'
        -> 'sma') and looks up the color for that type.

        Parameters
        ----------
        feature_name : str
            The name of the feature column (e.g., 'SMA_20').

        Returns
        -------
        List[float]
            A list of three float values for the RGB color, normalized to [0, 1].
        """
        fn_lower = feature_name.lower()
        if fn_lower in channel_mapping:
            return [c / 255.0 for c in channel_mapping[fn_lower]]
        
        prefix = feature_name.split('_')[0].upper()
        config_key = PREFIX_MAP.get(prefix)
        
        if config_key and config_key in channel_mapping:
            return [c / 255.0 for c in channel_mapping[config_key]]
        
        return [0.5, 0.5, 0.5]

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    
    # =============================================================================
    # Create ax_price first, then create ax_volume using ax_price as the
    # shared x-axis. This resolves potential UnboundLocalError issues.
    # =============================================================================
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
    
    for ax in [fig, ax_price, ax_volume]:
        ax.set_facecolor('black')
    
    min_price, max_price = window_df['low'].min(), window_df['high'].max()
    if pd.isna(min_price) or pd.isna(max_price):
        plt.close(fig)
        return None

    ax_price.set_ylim(min_price, max_price)
    ax_price.set_xlim(-1, lookback)

    oscillators_to_normalize = ['OBV', 'MACD', 'MACDH', 'MACDS', 'STOCHK', 'STOCHD', 'VTXP', 'VTXM', 'AO', 'RSI', 'ATRR']
    for col_name in window_df.columns:
        prefix = col_name.split('_')[0].upper()
        if prefix in oscillators_to_normalize:
            min_val, max_val = window_df[col_name].min(), window_df[col_name].max()
            if not pd.isna(min_val) and not pd.isna(max_val) and max_val > min_val:
                denominator = (max_val - min_val) + 1e-8
                window_df[col_name] = min_price + ((window_df[col_name] - min_val) * (max_price - min_price) / denominator)
            else:
                # If the indicator is flat, draw it in the middle.
                window_df[col_name] = (min_price + max_price) / 2

    # =============================================================================
    if all(k in window_df for k in ['open', 'high', 'low', 'close']):
        ohlc_color = get_color_for_feature('ohlc')
        for day_idx, row in enumerate(window_df.itertuples()):
            ax_price.plot([day_idx, day_idx], [row.low, row.high], color=ohlc_color, linewidth=1)
            ax_price.plot([day_idx, day_idx - 0.2], [row.open, row.open], color=ohlc_color, linewidth=1)
            ax_price.plot([day_idx, day_idx + 0.2], [row.close, row.close], color=ohlc_color, linewidth=1)

    # Draw all other indicator lines
    for col_name in window_df.columns:
        if col_name.lower() in ['open', 'high', 'low', 'close', 'volume']:
            continue
        line_color = get_color_for_feature(col_name)
        ax_price.plot(np.arange(lookback), window_df[col_name], color=line_color, linewidth=0.8)

    # Draw volume bars
    if 'volume' in window_df:
        max_volume = window_df['volume'].max()
        ax_volume.set_ylim(0, max_volume * 1.1 if max_volume > 0 else 1)
        vol_color = get_color_for_feature('volume')
        ax_volume.bar(np.arange(lookback), window_df['volume'], color=vol_color)

    # Finalize plot aesthetics
    for ax in [ax_price, ax_volume]:
        ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0)

    # Convert to NumPy array
    fig.canvas.draw()
    image_np = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    # Save the image as a compressed NumPy array
    npz_path = os.path.join(output_dir, f"image_{i}.npz")
    np.savez_compressed(npz_path, image=image_np)

    # Save an example PNG of the first image for easy viewing
    if i == 0:
        png_path = os.path.join(output_dir, "example_image.png")
        plt.imsave(png_path, image_np)
    
    return npz_path, label

def generate_images_from_df(df: pd.DataFrame, config: Dict[str, Any], output_dir: str, is_live_prediction: bool = False) -> Tuple[List[str], np.ndarray, pd.DatetimeIndex]:
    """Generates chart-like images from a time-series DataFrame.

    This function orchestrates the entire process of converting a feature-rich
    DataFrame into a collection of image samples for model training or
    prediction. It first applies a labeling strategy based on the
    configuration, then slides a window over the data, and for each window,
    it dispatches a task to `_create_and_save_image` for parallel processing.
    It handles both training/backtesting (with labels) and live prediction
    (without labels) modes.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing price data and all calculated
        technical indicators. Must have a datetime index.
    config : Dict[str, Any]
        The main configuration dictionary.
    output_dir : str
        The path to the directory where the generated .npz images will be saved.
    is_live_prediction : bool, optional
        If True, labels are not generated, and images are created for all
        possible data points up to the end of the DataFrame. Defaults to False.

    Returns
    -------
    Tuple[List[str], np.ndarray, pd.DatetimeIndex]
        A tuple containing three elements:
        - A list of file paths to the generated .npz images.
        - A NumPy array of the corresponding labels (or dummy labels if
          `is_live_prediction` is True).
        - A pandas DatetimeIndex with the end date for each image.

    Raises
    ------
    ValueError
        If any feature columns cannot be mapped to a color via the
        configuration, or if an unknown labeling strategy is specified.
    """
    print("\n--- Starting parallel image generation ---")

    # --- Validation Check ---
    print("Validating feature-to-color mapping...")
    channel_mapping = config['imaging']['channel_mapping']
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    
    unmapped_cols = []
    indicator_cols = [col for col in df.columns if col.lower() not in base_cols and 'label' not in col.lower() and 'future' not in col.lower()]

    for col in indicator_cols:
        prefix = col.split('_')[0].upper()
        config_key = PREFIX_MAP.get(prefix)
        if not config_key or config_key not in channel_mapping:
            unmapped_cols.append(col)
    
    if unmapped_cols:
        raise ValueError(
            f"Feature mapping error! The following {len(unmapped_cols)} feature(s) "
            f"could not be mapped to a color in your config: {unmapped_cols}\n"
            "Please check the `PREFIX_MAP` in `src/data/imager.py`."
        )
    print(f"✅ All {len(indicator_cols)} indicator columns successfully mapped to colors.")
    
    label_config = config['labeling']
    lookback = config['imaging']['lookback_period']
    horizon = label_config['horizon']

    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to '{output_dir}'")

    # Logic to distinguish between backtesting and live prediction
    if not is_live_prediction:
        print("Mode: Backtesting (generating labels)")
        df['future_close'] = df['close'].shift(-horizon)
        strategy_type = label_config['strategy']['type']
        print(f"Applying labeling strategy: '{strategy_type}'")

        # --- Apply the selected labeling strategy ---
        if strategy_type == "binary_classification":
            df['label'] = (df['future_close'] > df['close']).astype(int)
        elif strategy_type == "price_change_target":
            params = label_config['price_change_target']
            threshold = params['threshold_percent']
            df['label'] = (df['future_close'] >= df['close'] * (1 + threshold)).astype(int)
            print(f"  (Threshold: Price must rise >= {threshold:.2%})")
        elif strategy_type == "volatility_band":
            params = label_config['volatility_band']
            band = params['band_percent']
            stable_label = params['stable_label']
            unstable_label = params['unstable_label']
            lower_bound = df['close'] * (1 - band)
            upper_bound = df['close'] * (1 + band)
            is_stable = (df['future_close'] >= lower_bound) & (df['future_close'] <= upper_bound)
            df['label'] = np.where(is_stable, stable_label, unstable_label)
            print(f"  (Band: +/- {band:.2%}, Stable Label: {stable_label})")
        else:
            raise ValueError(f"Unknown labeling strategy type: {strategy_type}")

        # Remove rows for which no label could be created (due to the look-ahead horizon)
        df.dropna(subset=['future_close', 'label'], inplace=True)
        df['label'] = df['label'].astype(int)
    else:
        print("Mode: Live Prediction (generating images only, no labels)")
        # Create dummy labels (0) as they are needed for task creation,
        # but are not used in live prediction mode.
        df['label'] = 0

    # --- Parallel Task Preparation ---
    tasks = []
    num_samples = len(df) - lookback + 1
    if num_samples <= 0:
        print("Not enough data to generate images after filtering.")
        return [], np.array([]), pd.Index([])

    print(f"Preparing {num_samples} image generation tasks...")
    for i in range(num_samples):
        window_df = df.iloc[i: i + lookback]
        # The label is either the real one (backtest) or a dummy (live)
        label = df.iloc[i + lookback - 1]['label']
        tasks.append((i, window_df, label, config, output_dir))

    # --- Execution ---
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(_create_and_save_image, tasks), total=len(tasks), desc="Generating Images"))

    valid_results = [res for res in results if res is not None]
    image_paths = [res[0] for res in valid_results]
    labels = np.array([res[1] for res in valid_results])
    
    image_indices = [task[0] for task in tasks[:len(valid_results)]]
    dates = [df.index[i + lookback - 1] for i in image_indices]
    
    print(f"\nSuccessfully generated and saved {len(image_paths)} images.")
    if len(image_paths) > 0:
        print(f"An example image has been saved as '{os.path.join(output_dir, 'example_image.png')}'")
    
    # Return the paths, labels, and corresponding dates
    return image_paths, labels, pd.to_datetime(dates)