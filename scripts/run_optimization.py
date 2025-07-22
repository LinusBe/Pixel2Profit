import sys
import os
import torch
import numpy as np
import random
import argparse

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import PixelToProfitPipeline

def set_seed(seed: int = 42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to {seed}")

def main():
    """
    Main function to configure and run the hyperparameter optimization pipeline.
    """
    set_seed(42)

    parser = argparse.ArgumentParser(description='Pixel-to-Profit: CNN Hyperparameter Optimization')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to the configuration file for the optimization study.'
    )
    args = parser.parse_args()

    # Initialize and run the pipeline
    pipeline = PixelToProfitPipeline(args.config)
    pipeline.run_optimization()

if __name__ == "__main__":
    main()