Of course. Here is the concise README in English, covering all the core functions of your pipeline.

-----

# Pixel-to-Profit: An Image-Based Financial Forecasting Pipeline

An end-to-end pipeline for developing trading strategies by converting time-series data into images to train Convolutional Neural Networks (CNNs).

-----

## Core Features

  * **Configuration-Driven**: The entire workflow is controlled via a single `.yaml` configuration file.
  * **Dual Optimization Modes**: Leverages Optuna for hyperparameter optimization, targeting either ML metrics (e.g., F1-Score) or financial KPIs (e.g., Sharpe Ratio).
  * **Robustness Analysis**: Features integrated Purged K-Fold Cross-Validation to validate the top-performing models.
  * **Advanced Backtesting**: Utilizes `vectorbt` for high-speed backtests, including a Monte Carlo benchmark and detailed trade analysis.
  * **Model Interpretability**: Implements SHAP and Permutation Importance to explain the decisions of the CNN.
  * **Scientific Workflow**: A fully automated, multi-stage analysis process from optimization to a final hold-out evaluation.

-----

## The Workflow

The pipeline follows a three-phase scientific process:

1.  **Phase 1: Optimization (`run_pipeline.py`)**

      * Executes multiple Optuna studies to find the best hyperparameter sets for various objective metrics. Each study is saved to a unique directory under `/runs`.

2.  **Phase 2: Robustness Analysis (`master_analyzer.py`)**

      * Identifies the top models from all completed studies.
      * Performs a rigorous Purged K-Fold Cross-Validation on these models.
      * Creates a "Golden Record" (`paper_results/golden_record...csv`) of the most robust candidates.

3.  **Phase 3: Final Evaluation (`run_final_evaluation.py`)**

      * Takes the robust models from the "Golden Record".
      * Runs a final backtest on an unseen hold-out test set.
      * Generates a final summary table comparing the performance of all top models.

-----

## How to Run

#### 1\. Start an Optimization

Run an optimization study to find the best models.

```bash
# Optimize for financial metrics (e.g., Sharpe Ratio)
python scripts/run_pipeline.py --mode financial_optimization --config configs/base_config.yaml

# Optimize for model metrics (e.g., F1-Score)
python scripts/run_pipeline.py --mode model_optimization --config configs/base_config.yaml
```

#### 2\. Run the Cross-Study Analysis

After several studies are complete, analyze the results to find the most robust models.

```bash
# (First, update the study paths inside the script)
python scripts/master_analyzer.py
```

#### 3\. Run the Final Evaluation

Execute the final backtest for all robust models on the hold-out data.

```bash
python scripts/run_final_evaluation.py --evaluate-all
```