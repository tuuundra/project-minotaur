#!/usr/bin/env python
"""
Train a CNN-Transformer model on pre-calculated features, potentially adding more.
This version (model_4_30) focuses on integrating Experiment E (daily regime)
and unifying feature selection logic.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timezone # Ensure timezone is imported
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib
import glob
from sklearn.isotonic import IsotonicRegression
import io
from sklearn.utils import class_weight as sk_class_weight
# +++ Add sklearn metrics for F1 callback +++
from sklearn.metrics import f1_score, precision_recall_curve
import talib
import numpy as np # Ensure numpy is imported
import json        # Ensure json is imported
import os          # Ensure os is imported
import time        # Ensure time is imported
import optuna # +++ ADD OPTUNA IMPORT +++
import re # For prepare_output_dir regex

# --- WandB Optuna Integration ---
# from optuna.integration import WandbCallback # For per-trial logging to WandB # TEMP COMMENT OUT FOR SMOKE TEST
# --- End WandB Optuna Integration ---

# +++ TensorFlow/Keras imports +++
import tensorflow as tf

# +++ ENABLE NUMERIC CHECKING FOR DEBUGGING +++
# tf.debugging.enable_check_numerics() # DISABLED FOR PERFORMANCE
# logger = logging.getLogger("tf_debug") # Use a specific logger for this message # DISABLED
# logger.info("TensorFlow numeric checking (tf.debugging.enable_check_numerics()) has been ENABLED.") # DISABLED
# +++ END NUMERIC CHECKING +++

from tensorflow.keras.models import Sequential, Model # Need Model for functional API
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    LayerNormalization, MultiHeadAttention, Add, Flatten, BatchNormalization,
    SpatialDropout1D, Concatenate, Layer, Embedding, multiply, Activation, Softmax, Lambda, # Added Softmax, SpatialDropout1D, Layer, etc.
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint # Added ModelCheckpoint
from tensorflow.keras.optimizers import AdamW # Use AdamW (moved from experimental)
from tensorflow.keras import regularizers # <-- Added import
import tensorflow.keras.backend as K # For attention pooling
from tensorflow.keras.utils import plot_model # <-- Added import for model visualization

# Add project root to path for imports if needed
current_dir = Path(__file__).parent
# Assuming chimera_5_19_optuna directory is at the root of the project
project_root = current_dir.parent 
sys.path.append(str(project_root))
# logger.info(f"Project root added to sys.path: {project_root}") # Logging this here might be too early if logger isn't fully set up

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Project root ({project_root}) added to sys.path.") # Log after logger is configured

# +++ Add NpEncoder for robust JSON serialization +++
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path): # Handling Path objects
            return str(obj)
        return super(NpEncoder, self).default(obj)
# +++ End NpEncoder +++

# --- Helper: Parse Arguments ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Single-Stack CNN-Transformer model for financial forecasting using Optuna.")

    # --- Data parameters ---
    parser.add_argument('--feature-parquet', type=str,
                        default="minotaur/data/consolidated_features_targets_all.parquet", 
                        help='Path to the pre-calculated feature Parquet file with targets.')
    parser.add_argument('--feature-list-file', type=str, default=None, # +++ NEW ARGUMENT +++
                        help='Path to a .txt file containing a list of feature names to use (one per line). Overrides --use-xxx-term-features flags.')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (chronological split)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Proportion of training data to use for validation (from the end of train set)')

    # --- Smoke Test Parameter ---
    parser.add_argument('--smoke-test-nrows', type=int, default=None,
                        help='If set, load only the first N rows from the feature Parquet for a quick smoke test.')

    # --- Feature Branch Selection (determines which features are concatenated for the single stack) ---
    parser.add_argument('--use-short-term-features', action='store_true', default=True,
                        help='Include short-term features in the model input.')
    parser.add_argument('--use-medium-term-features', action='store_true', default=True,
                        help='Include medium-term features in the model input.')
    parser.add_argument('--use-long-term-features', action='store_true', default=False,
                        help='Include long-term features in the model input.')
    parser.add_argument('--use-regime-features', action='store_true', default=False, 
                        help='Include regime-based features in the model input.')

    # --- Normalization parameters ---
    parser.add_argument('--normalization', type=str, default='robust',
                        choices=['standard', 'minmax', 'robust', 'power_yeo_johnson', 
                                 'quantile_normal', 'quantile_uniform', 'none'],
                        help='Feature normalization method for pre-processing (or none). Default changes to \'none\' if --use-instance-norm is active.')

    # --- Instance Normalization (RevIN-like) Parameters ---
    parser.add_argument('--use-instance-norm', action=argparse.BooleanOptionalAction, default=True,
                        help="Enable instance normalization (RevIN-like layer) at the model input.")
    parser.add_argument('--instance-norm-affine', action=argparse.BooleanOptionalAction, default=True,
                        help="Use learnable affine parameters (gamma/beta) in the instance normalization layer.")

    # --- Model Parameters (Defaults/Fixed - Optuna will primarily control these) ---
    parser.add_argument('--sequence-length', type=int, default=60, help='Length of input sequences for the model.')
    # Arguments like se-ratio, specific cnn-filters/kernels/dilations per branch are removed.
    # Optuna will handle the single CNN stack configuration.
    parser.add_argument('--num-transformer-blocks', type=int, default=2, help='(Default/Fallback) Number of Transformer encoder blocks if not tuned.') # Optuna: "blocks" o3: 1-3
    parser.add_argument('--num-heads', type=int, default=2, help='(Default/Fallback) Number of attention heads if not tuned.') # Optuna: "heads" o3: 1-4
    parser.add_argument('--head-size', type=int, default=64, help='(Default/Fallback) Dimensionality of each attention head if not tuned.') # Optuna: "head" o3: [32,48,64,96]
    parser.add_argument('--ff-dim-factor', type=float, default=2.0, help='(Default/Fallback) Factor for transformer feed-forward dim if not tuned.') # Optuna: "ff" o3: [2.0, 4.0]
    parser.add_argument('--mlp-units', type=int, nargs='+', default=[64, 32], help='(Default/Fallback) MLP layers units if not tuned.')
    parser.add_argument('--dropout-rate', type=float, default=0.30, help='(Default/Fallback) General dropout rate for Transformer if not tuned.') # Optuna: "drop" o3: 0.15-0.40
    parser.add_argument('--mlp-dropout-rate', type=float, default=0.35, help='(Default/Fallback) Dropout rate for MLP head if not tuned.') # Tied to "drop"
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='(Default/Fallback) Base learning rate if not tuned.') # Optuna: "lr" o3: 5e-6 to 5e-5
    parser.add_argument('--l2_strength', type=float, default=1e-3, help='L2 regularization strength (fixed for Optuna runs as per o3).') # Not tuned by Optuna per o3
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='(Default/Fallback) Base weight decay for AdamW if not tuned.') # Optuna: "wd" o3: 5e-5 to 5e-4
    parser.add_argument('--cnn-activation', type=str, default='gelu', choices=['relu', 'gelu', 'swish'], 
                        help='Activation for CNN layers. Fixed to "gelu" in Optuna runs as per o3.') # Fixed to gelu for Optuna
    parser.add_argument('--cnn-no-bias', action='store_true', help='Disable bias term in Conv1D layers.')

    # --- Training parameters ---
    parser.add_argument('--epochs', type=int, default=100, help='(Default) Maximum number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='(Default/Fallback) Batch size for training if not tuned.') # Optuna: "batch"
    parser.add_argument('--batch-size-eval', type=int, default=None, 
                        help='Batch size for evaluation. Defaults to training batch_size.')
    parser.add_argument('--steps-per-epoch', type=int, default=None, help='Limit steps per epoch for quick testing.')
    parser.add_argument('--use-class-weights', action='store_true', default=True, help='Use class weights. Defaulting to True as per o3 latest recommendation.') # o3 latest: ENABLE
    parser.add_argument('--early-stopping-patience', type=int, default=4, help='Patience for early stopping on val_auc (o3: 4).') # o3: 4
    parser.add_argument('--reduce-lr-patience', type=int, default=5, help='Patience for reducing LR on val_auc plateau.') 
    parser.add_argument('--reduce-lr-factor', type=float, default=0.2, help='Factor to reduce LR by.')
    parser.add_argument('--f1-callback-patience', type=int, default=0, help='Patience for F1EvalCallback early stopping (0 to disable).')

    # --- Class Weight Customization ---
    parser.add_argument('--class-weight-balanced', action='store_true', default=False,
                        help="Use 'balanced' heuristic for class weights.")
    parser.add_argument('--class-weight-0', type=float, default=1.0, help='Custom weight for class 0.')
    parser.add_argument('--class-weight-1', type=float, default=1.0, help='Custom weight for class 1.')

    # --- Focal Loss Parameters ---
    parser.add_argument('--use-focal-loss', action='store_true', default=True, help="Use Focal Loss. Defaulting to True.") # MODIFIED default to True
    parser.add_argument('--focal-alpha', type=float, default=0.25, help='(Default/Fallback) Alpha for Focal Loss if not tuned.') # Optuna: "f_alpha"
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='(Default/Fallback) Gamma for Focal Loss if not tuned.') # Optuna: "f_gamma"
    parser.add_argument('--label-smoothing', type=float, default=0.0, help="Label smoothing amount (o3: 0.0).") # o3: 0.0
    parser.add_argument('--clip-features', action='store_true', help="Clip features based on training data quantiles.")
    parser.add_argument('--clip-lower-q', type=float, default=0.01, help='Lower quantile for clipping.')
    parser.add_argument('--clip-upper-q', type=float, default=0.99, help='Upper quantile for clipping.')

    # --- WandB Integration ---
    parser.add_argument('--use-wandb', action='store_true', default=False, help='Enable WandB logging.')
    parser.add_argument('--wandb-project', type=str, default='optuna-chimera-single-stack', 
                        help='WandB project name.')
    parser.add_argument('--wandb-entity', type=str, default=None, help='WandB entity name.')

    # --- Output parameters ---
    parser.add_argument('--output-dir', type=str, default="minotaur/optuna_studies", 
                        help="Base directory for Optuna study runs.") # MODIFIED default path
    parser.add_argument('--plot-name-suffix', type=str, default="", help="Suffix for plot filenames.")
    parser.add_argument('--skip-plots', action='store_true', help="Skip model plot generation.")
    parser.add_argument('--save-results', action=argparse.BooleanOptionalAction, default=True, 
                        help="Save all results (models, plots, metrics). Use --no-save-results to disable.")

    # --- Keras Training Verbosity ---
    parser.add_argument('--keras-verbose', type=int, default=1,
                        choices=[0, 1, 2], help='Keras model.fit verbose setting.')

    # --- Optuna specific parameters ---
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials (o3: 40).') # o3: 40
    parser.add_argument('--optuna-db-filename', type=str, default='optuna_study_single_stack.db', 
                        help='Optuna storage database filename. Will be placed in the run-specific output directory.')
    parser.add_argument('--optuna-study-name', type=str, default='chimera_single_stack_optimization', 
                        help='Name for the Optuna study.')
    parser.add_argument('--optuna-direction', type=str, default='maximize', choices=['minimize', 'maximize'], 
                        help='Optuna optimization direction.')

    # --- Symbol ---
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol for naming output directory.')

    # +++ Fixed Hyperparameters for --no-optuna mode +++
    parser.add_argument('--no-optuna', action='store_true',
                        help='Run a single trial with fixed hyperparameters instead of an Optuna study.')
    # CNN
    parser.add_argument('--cnn-filters-1-fixed', type=int, default=64, help='Fixed CNN filters for layer 1 (if --no-optuna).')
    parser.add_argument('--cnn-kernel-size-1-fixed', type=int, default=5, help='Fixed CNN kernel size for layer 1 (if --no-optuna).')
    # cnn_filters_2, kernel_2, etc. would be added here if model supports more tuned CNN layers
    parser.add_argument('--cnn-pool-size-fixed', type=int, default=2, help='Fixed CNN pooling size (if --no-optuna).')
    parser.add_argument('--cnn-pool-strides-fixed', type=int, default=2, help='Fixed CNN pooling strides (if --no-optuna).')
    parser.add_argument('--cnn-dropout-rate-fixed', type=float, default=0.20, help='Fixed CNN dropout rate (if --no-optuna).')


    # Embedding
    parser.add_argument('--cat-embedding-dim-factor-fixed', type=float, default=0.5, help='Fixed categorical embedding dim factor (if --no-optuna).')

    # Transformer
    parser.add_argument('--num-transformer-blocks-fixed', type=int, default=None, help='Fixed number of Transformer blocks (if --no-optuna). Uses args.num_transformer_blocks default if None.')
    parser.add_argument('--num-heads-fixed', type=int, default=None, help='Fixed number of attention heads (if --no-optuna). Uses args.num_heads default if None.')
    parser.add_argument('--head-size-fixed', type=int, default=None, help='Fixed dimensionality of each attention head (if --no-optuna). Uses args.head_size default if None.')
    parser.add_argument('--ff-dim-factor-fixed', type=float, default=None, help='Fixed factor for transformer feed-forward dim (if --no-optuna). Uses args.ff_dim_factor default if None.')
    
    # MLP
    parser.add_argument('--mlp-units-fixed', type=str, default="64,32", help='Fixed MLP units as comma-separated string e.g., "64,32" (if --no-optuna).')
    
    # General Training / Regularization
    parser.add_argument('--dropout-rate-fixed', type=float, default=None, help='Fixed general dropout rate (if --no-optuna). Uses args.dropout_rate default if None.')
    parser.add_argument('--mlp-dropout-rate-fixed', type=float, default=None, help='Fixed MLP dropout rate (if --no-optuna). Uses args.mlp_dropout_rate default if None.')
    parser.add_argument('--learning-rate-fixed', type=float, default=None, help='Fixed learning rate (if --no-optuna). Uses args.learning_rate default if None.')
    parser.add_argument('--weight-decay-fixed', type=float, default=None, help='Fixed weight decay for AdamW (if --no-optuna). Uses args.weight_decay default if None.')
    parser.add_argument('--batch-size-fixed', type=int, default=None, help='Fixed batch size for training (if --no-optuna). Uses args.batch_size default if None.')

    # Loss
    parser.add_argument('--focal-alpha-fixed', type=float, default=None, help='Fixed alpha for Focal Loss (if --no-optuna). Uses args.focal_alpha default if None.')
    parser.add_argument('--focal-gamma-fixed', type=float, default=None, help='Fixed gamma for Focal Loss (if --no-optuna). Uses args.focal_gamma default if None.')
    
    # Instance Norm
    parser.add_argument('--instance-norm-affine-fixed', action=argparse.BooleanOptionalAction, default=True, help="Fixed: Use learnable affine parameters in instance norm (if --no-optuna).")

    # +++ NEW: Max samples for NumPy evaluation sequences +++
    parser.add_argument('--max-samples-for-np-eval', type=int, default=100000, # Default to 100k samples
                        help='Maximum number of samples from the end of val/test sets to use for creating NumPy sequence arrays for evaluation (isotonic calibration, final test metrics). Helps manage memory for large datasets.')

    parsed_args = parser.parse_args()

    if parsed_args.batch_size_eval is None:
        parsed_args.batch_size_eval = parsed_args.batch_size
        logger.info(f"Batch size for evaluation (--batch-size-eval) set to training batch size: {parsed_args.batch_size_eval}")

    # If instance normalization is used, set default pre-processing normalization to 'none'
    if parsed_args.use_instance_norm and parsed_args.normalization == 'robust': # Only override if it's still the original default
        logger.info("Instance normalization (--use-instance-norm) is active. "
                    "Overriding default pre-processing normalization from 'robust' to 'none'.")
        parsed_args.normalization = 'none'
    elif parsed_args.use_instance_norm :
        logger.info(f"Instance normalization (--use-instance-norm) is active. "
                    f"Pre-processing normalization is set to '{parsed_args.normalization}'.")

    # Populate fixed args from their respective primary args if None (for --no-optuna mode)
    if parsed_args.num_transformer_blocks_fixed is None:
        parsed_args.num_transformer_blocks_fixed = parsed_args.num_transformer_blocks
    if parsed_args.num_heads_fixed is None:
        parsed_args.num_heads_fixed = parsed_args.num_heads
    if parsed_args.head_size_fixed is None:
        parsed_args.head_size_fixed = parsed_args.head_size
    if parsed_args.ff_dim_factor_fixed is None:
        parsed_args.ff_dim_factor_fixed = parsed_args.ff_dim_factor
    # mlp_units_fixed has its own default string
    if parsed_args.dropout_rate_fixed is None:
        parsed_args.dropout_rate_fixed = parsed_args.dropout_rate
    if parsed_args.mlp_dropout_rate_fixed is None:
        parsed_args.mlp_dropout_rate_fixed = parsed_args.mlp_dropout_rate
    if parsed_args.learning_rate_fixed is None:
        parsed_args.learning_rate_fixed = parsed_args.learning_rate
    if parsed_args.weight_decay_fixed is None:
        parsed_args.weight_decay_fixed = parsed_args.weight_decay
    if parsed_args.batch_size_fixed is None:
        parsed_args.batch_size_fixed = parsed_args.batch_size
    if parsed_args.focal_alpha_fixed is None:
        parsed_args.focal_alpha_fixed = parsed_args.focal_alpha
    if parsed_args.focal_gamma_fixed is None:
        parsed_args.focal_gamma_fixed = parsed_args.focal_gamma
    # instance_norm_affine_fixed has its own default action=argparse.BooleanOptionalAction

    return parsed_args

# --- Helper: Setup Output Directory ---
def prepare_output_dir(base_output_dir_for_this_trial_set, symbol, trial_number=None):
    """Creates a unique run directory for a specific trial's artifacts."""
    base_path = Path(base_output_dir_for_this_trial_set) 
    clean_symbol = symbol.replace('/', '_').replace(':', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    run_prefix_base = f"{clean_symbol}_cnntransformer_single_stack"

    if trial_number is not None:
        run_dir_name = f"trial_{trial_number:03d}_{run_prefix_base}_{timestamp}"
    else:
        numeric_prefix_pattern = re.compile(r"^(\\d+)_.*") 
        max_run_num = 0
        if base_path.exists() and base_path.is_dir(): 
            for p_str in os.listdir(base_path):
                p = base_path / p_str
                if p.is_dir():
                    dir_name = p.name
                    if not dir_name.startswith("trial_"): 
                        match = numeric_prefix_pattern.match(dir_name)
                        if match:
                            try:
                                run_num = int(match.group(1))
                                if run_num > max_run_num:
                                    max_run_num = run_num
                            except ValueError:
                                pass
        next_run_num = max_run_num + 1 
        run_prefix_num_str = f"{next_run_num:03d}"
        run_dir_name = f"{run_prefix_num_str}_{run_prefix_base}_{timestamp}"

    final_trial_artifact_dir = base_path / run_dir_name 
    try:
        os.makedirs(final_trial_artifact_dir, exist_ok=True)
        logger.info(f"Trial artifacts directory: {final_trial_artifact_dir}")
    except OSError as e:
        logger.error(f"Error creating trial artifacts directory {final_trial_artifact_dir}: {e}")
        raise
    return final_trial_artifact_dir

# --- Constants ---
TARGET_COLUMN_NAME = 'target_tp2.0_sl1.0' # New constant for our target
# --- End Constants ---

# --- Import TA-Lib explicitly if needed for target calculation ---
try:
    import talib as ta
    TALIB_AVAILABLE = True
    logger.info("TA-Lib imported successfully.")
except ImportError:
    ta = None
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not found. Target calculation might fail if needed.")


# --- Copy necessary target calculation functions directly ---

# +++ Target Definition Function (Adaptive SL/TP - Copied from simple_model_train_v5.py) +++
def create_adaptive_sl_tp_target(data, min_sl_pct=1.0, atr_period=14, atr_multiplier=1.5):
    # ... (Keep the exact implementation of create_adaptive_sl_tp_target from v5) ...
    """
    Calculates target based on adaptive SL (min % or ATR) and 2:1 TP, no time limit.

    Args:
        data (pd.DataFrame): DataFrame with at least ['high', 'low', 'close'] and 'atr_XX' if not recalculating ATR.
                             If 'atr_XX' (e.g., 'atr_14') is not present, it will be calculated.
        min_sl_pct (float): Minimum stop loss percentage below entry price.
        atr_period (int): Period for ATR calculation.
        atr_multiplier (float): Multiplier for ATR component of stop loss.

    Returns:
        np.ndarray: Array of targets (0, 1, or np.nan)
    """
    required_cols = ['high', 'low', 'close']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Dataframe must contain {required_cols} columns for adaptive target.")

    # --- Check if ATR needs to be calculated or use existing ---
    atr_col_name = f'atr_{atr_period}' # e.g., 'atr_14'
    if atr_col_name not in data.columns:
        logger.info(f"'{atr_col_name}' not found in DataFrame. Calculating ATR({atr_period})...")
        if not TALIB_AVAILABLE:
            raise ImportError("TA-Lib is required for adaptive target calculation but not found.")
        try:
            # Use lowercase column names for TA-Lib calculation
            high_prices = data['high'].values.astype(np.float64)
            low_prices = data['low'].values.astype(np.float64)
            close_prices = data['close'].values.astype(np.float64)

            atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=atr_period)
            atr_series = pd.Series(atr, index=data.index) # Convert numpy array back to Series with original index
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}. Ensure TA-Lib is installed correctly and inputs are valid.")
            raise
    else:
        logger.info(f"Using existing '{atr_col_name}' column for adaptive target calculation.")
        atr_series = data[atr_col_name]
    # --- End ATR Check ---

    logger.info(f"Calculating adaptive 2:1 targets (Min SL={min_sl_pct}%, ATR Period={atr_period}, ATR Mult={atr_multiplier})...")

    targets = np.full(len(data), np.nan) # Initialize with NaN

    first_valid_idx_label = atr_series.dropna().index.min()
    if pd.isna(first_valid_idx_label):
        logger.warning("ATR calculation resulted in all NaNs. Cannot create adaptive targets.")
        return targets

    try:
        first_valid_pos = data.index.get_loc(first_valid_idx_label)
    except KeyError:
         logger.warning(f"Could not find index label {first_valid_idx_label} in DataFrame index. Searching linearly.")
         first_valid_pos = -1
         for i, val in enumerate(atr_series):
             if not pd.isna(val):
                 first_valid_pos = i
                 break
         if first_valid_pos == -1:
             logger.warning("No valid ATR values found even with linear search. Cannot create adaptive targets.")
             return targets

    logger.info(f"Starting adaptive target calculation loop from index position {first_valid_pos}...")

    # Use lowercase column names here
    close_values = data['close'].values
    low_values = data['low'].values
    high_values = data['high'].values
    atr_values = atr_series.values

    for i_pos in range(first_valid_pos, len(data) - 1):
        entry_price = close_values[i_pos]
        atr_value = atr_values[i_pos]

        if pd.isna(entry_price) or entry_price <= 0 or pd.isna(atr_value):
            continue

        distance_pct = entry_price * (min_sl_pct / 100.0)
        distance_atr = atr_multiplier * atr_value
        stop_loss_distance = max(distance_pct, distance_atr)

        if stop_loss_distance <= 0 or stop_loss_distance >= entry_price:
            continue

        stop_loss_price = entry_price - stop_loss_distance
        take_profit_price = entry_price + 2.0 * stop_loss_distance

        future_lows = low_values[i_pos + 1:]
        future_highs = high_values[i_pos + 1:]

        target_hit = False
        for j in range(len(future_lows)):
            if future_lows[j] <= stop_loss_price:
                targets[i_pos] = 0
                target_hit = True
                break
            if future_highs[j] >= take_profit_price:
                targets[i_pos] = 1
                target_hit = True
                break
        if not target_hit:
            targets[i_pos] = np.nan # Explicitly set NaN if neither SL nor TP is hit

    logger.info("Adaptive target calculation complete.")
    return targets

# +++ Add other target functions (fixed_horizon, fixed_pct) if needed +++
def create_fixed_horizon_target(data, horizon=1):
    # ... (Keep implementation) ...
    logger.info(f"Calculating fixed horizon target with horizon={horizon}...")
    # Use lowercase column name
    future_close = data['close'].shift(-horizon)
    close_safe = data['close'].replace(0, np.nan)
    returns = (future_close - close_safe) / close_safe
    target = (returns > 0).astype(float)
    target[returns.isna()] = np.nan
    return target

def create_fixed_pct_sl_tp_target(data, tp_pct=0.5, sl_pct=0.25):
    # ... (Keep implementation) ...
    required_cols = ['high', 'low', 'close'] # Use lowercase
    if not all(col in data.columns for col in required_cols):
         raise ValueError(f"Dataframe must contain {required_cols} columns for fixed % target.")

    logger.info(f"Calculating fixed % SL/TP targets (TP={tp_pct}%, SL={sl_pct}%)...")
    targets = np.full(len(data), np.nan)
    # Use lowercase column names
    close_values = data['close'].values
    low_values = data['low'].values
    high_values = data['high'].values

    for i in range(len(data) - 1):
        entry_price = close_values[i]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        stop_loss_price = entry_price * (1.0 - sl_pct / 100.0)
        take_profit_price = entry_price * (1.0 + tp_pct / 100.0)

        future_lows = low_values[i + 1:]
        future_highs = high_values[i + 1:]

        target_hit = False
        for j in range(len(future_lows)):
            if future_lows[j] <= stop_loss_price:
                targets[i] = 0
                target_hit = True
                break
            if future_highs[j] >= take_profit_price:
                targets[i] = 1
                target_hit = True
                break
        # Removed the explicit NaN setting here, default is NaN if loop finishes

    logger.info("Fixed % SL/TP target calculation complete.")
    return targets
# --- End Copied Target Functions ---


# --- Focal Loss Implementation ---

def create_focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.0):
    """Wrapper to create a focal loss function with given parameters."""
    # This function now correctly calls the existing focal_loss which returns the actual loss fn
    return focal_loss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)

# ... (Keep implementation) ...
def focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.0):
    # ... (Implementation) ...
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1.0 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1.0 - alpha_factor)
        modulating_factor = K.pow((1.0 - p_t), gamma)
        loss = alpha_t * modulating_factor * cross_entropy
        return K.mean(loss, axis=-1)
    return focal_loss_fixed

# +++ Custom Safe F1 Score Metric +++
class SafeF1Score(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='f1_score', dtype=tf.float32):
        super(SafeF1Score, self).__init__(name=name, dtype=dtype)
        self.threshold = threshold
        # Initialize state variables to store counts
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply threshold to get binary predictions
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, self.dtype))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, self.dtype))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), self.dtype))

        # Update state variables
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

        # Note: sample_weight is ignored in this basic implementation for simplicity,
        # but could be incorporated by multiplying counts if needed.

    def result(self):
        # Calculate precision and recall, guarding against division by zero
        precision = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = tf.math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

        # Calculate F1 score, guarding against division by zero
        f1 = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return f1

    def reset_state(self):
        # Reset state variables
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.false_negatives.assign(0.0)

    def get_config(self):
        # Ensure threshold is serializable
        config = super(SafeF1Score, self).get_config()
        config.update({'threshold': float(self.threshold)}) # Cast to float
        return config
# +++ End Custom Safe F1 Score Metric +++

# +++ Sequence Creation Function +++
# ... (Keep implementation) ...
def create_sequences(data, target, sequence_length):
    # ... (Implementation) ...
    num_samples = len(data)
    
    if num_samples < sequence_length:
        logger.warning(f"Data length ({num_samples}) is less than sequence length ({sequence_length}). Cannot create sequences.")
        # Determine num_features safely, even for 1D data (though typically features are 2D)
        num_features = data.shape[1] if data.ndim > 1 else (1 if data.ndim == 1 and data.size > 0 else 0)
        # If data.ndim is 0 (scalar) or data is empty, num_features might be problematic,
        # but this case should ideally be caught by num_samples < sequence_length or result in 0 sequences.
        # For robustness, ensure num_features is sensible for empty sequence array shape.
        if num_features == 0 and sequence_length > 0 : # if data was truly empty or scalar
             logger.warning(f"Input data has 0 features or is empty, sequence_length={sequence_length}.")

        # Return empty arrays with correct dimensionality for sequences
        # sequences should be (0, sequence_length, num_features)
        # targets should be (0,)
        return np.empty((0, sequence_length, num_features if num_features > 0 else 1 if sequence_length > 0 else 0), dtype=data.dtype), np.empty((0,), dtype=target.dtype)

    num_sequences_possible = num_samples - sequence_length + 1
    
    # Determine num_features from the data
    if data.ndim == 1: # Handle case where input data might be 1D (e.g. a single feature)
        num_features = 1
        # Reshape data to be 2D for consistent sequence creation: (num_samples, 1)
        data_for_sequencing = data.reshape(-1, 1)
    elif data.ndim > 1:
        num_features = data.shape[1]
        data_for_sequencing = data
    else: # data.ndim == 0 (scalar), should have been caught by num_samples < sequence_length or result in 0 sequences
        logger.error(f"Input data has unexpected ndim: {data.ndim}. Shape: {data.shape}")
        # Fallback, though this path is unlikely if initial checks pass
        return np.empty((0, sequence_length, 1), dtype=data.dtype), np.empty((0,), dtype=target.dtype)

    # Pre-allocate NumPy arrays
    # Ensure dtype is preserved from original data and target arrays
    output_sequences = np.empty((num_sequences_possible, sequence_length, num_features), dtype=data_for_sequencing.dtype)
    output_targets = np.empty(num_sequences_possible, dtype=target.dtype)

    for i in range(num_sequences_possible):
        output_sequences[i] = data_for_sequencing[i:i + sequence_length]
        output_targets[i] = target[i + sequence_length - 1]
        
    return output_sequences, output_targets

# --- Data Sequence Generator for tf.data.Dataset ---
def data_sequence_generator(features_df, targets_series, sequence_length, batch_size, shuffle=True):
    """
    Generates batches of sequences and corresponding targets.

    Args:
        features_df (pd.DataFrame): DataFrame of input features.
        targets_series (pd.Series): Series of target labels.
        sequence_length (int): The length of each sequence.
        batch_size (int): The number of sequences per batch.
        shuffle (bool): Whether to shuffle the data before generating batches (typically True for training).
    
    Yields:
        tuple: A tuple containing (batch_X, batch_y)
               batch_X (np.ndarray): Batch of sequences, shape (batch_size, sequence_length, n_features), dtype float32.
               batch_y (np.ndarray): Batch of targets, shape (batch_size, 1), dtype int32.
    """
    num_samples = len(features_df)
    if num_samples == 0:
        logger.warning("data_sequence_generator received empty features_df. Yielding nothing.")
        return

    # Calculate the number of possible sequences. 
    # Adjusted for 0-based indexing: if num_samples = 60, seq_len = 60, then 1 sequence possible (indices 0-59)
    # last_possible_start_index = num_samples - sequence_length
    # num_possible_sequences = last_possible_start_index + 1
    num_possible_sequences = num_samples - sequence_length + 1

    if num_possible_sequences <= 0:
        logger.warning(f"data_sequence_generator: Not enough data ({num_samples} samples) for sequence length {sequence_length}. Yielding nothing.")
        return

    indices = np.arange(num_possible_sequences) # These are the starting indices of sequences
    if shuffle:
        np.random.shuffle(indices)

    num_batches = num_possible_sequences // batch_size
    if num_batches == 0 and num_possible_sequences > 0: # if less than one full batch, still create one partial batch
        num_batches = 1 
        # logger.debug(f"data_sequence_generator: num_batches was 0, set to 1 for {num_possible_sequences} sequences and batch_size {batch_size}")
    elif num_batches == 0 and num_possible_sequences == 0:
        logger.warning("data_sequence_generator: num_batches is 0 and no possible sequences. Yielding nothing.")
        return # No data to yield

    # logger.debug(f"data_sequence_generator: num_samples={num_samples}, seq_len={sequence_length}, num_possible_sequences={num_possible_sequences}, batch_size={batch_size}, num_batches={num_batches}")

    for i in range(num_batches):
        batch_indices_start = indices[i * batch_size : (i + 1) * batch_size]
        current_batch_size = len(batch_indices_start)

        # Initialize batch data containers
        # Assuming features_df is (num_samples, num_features)
        # batch_X_data will be (current_batch_size, sequence_length, num_features)
        batch_X_data = np.zeros((current_batch_size, sequence_length, features_df.shape[1]), dtype=np.float32) 
        batch_y_data = np.zeros((current_batch_size, 1), dtype=np.int32) # Assuming target is int32

        for j, start_idx in enumerate(batch_indices_start):
            end_idx = start_idx + sequence_length
            # features_df is already a NumPy array, use direct slicing
            batch_X_data[j] = features_df[start_idx:end_idx].astype(np.float32)
            # targets_series is also a NumPy array. The target is for the END of the sequence.
            batch_y_data[j] = targets_series[end_idx - 1].astype(np.int32) 
        
        # logger.debug(f"Yielding batch X shape: {batch_X_data.shape}, y shape: {batch_y_data.shape}")
        yield batch_X_data, batch_y_data
    
    # Handle the last partial batch if shuffle is True or if explicitly needed
    # If not shuffling, typically steps_per_epoch handles not using partials unless specifically designed
    # However, for completeness, especially if shuffle=False and we want all data:
    remaining_sequences = num_possible_sequences % batch_size
    if remaining_sequences > 0 and i == num_batches -1: # only if there was at least one full batch processed and there's a remainder
        # This part might be redundant if steps_per_epoch is calculated as floor division
        # and model.fit handles it correctly. This is more for manual iteration.
        batch_indices_start_partial = indices[num_batches * batch_size:]
        current_partial_batch_size = len(batch_indices_start_partial)

        if current_partial_batch_size > 0:
            # logger.debug(f"Handling partial batch of size {current_partial_batch_size}")
            batch_X_data_partial = np.zeros((current_partial_batch_size, sequence_length, features_df.shape[1]), dtype=np.float32)
            batch_y_data_partial = np.zeros((current_partial_batch_size, 1), dtype=np.int32)

            for j, start_idx in enumerate(batch_indices_start_partial):
                end_idx = start_idx + sequence_length
                batch_X_data_partial[j] = features_df[start_idx:end_idx].astype(np.float32)
                batch_y_data_partial[j] = targets_series[end_idx - 1].astype(np.int32)
            
            # logger.debug(f"Yielding partial batch X shape: {batch_X_data_partial.shape}, y shape: {batch_y_data_partial.shape}")
            yield batch_X_data_partial, batch_y_data_partial

# +++ Transformer Block Definition +++
# ... (Keep implementation) ...
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0, ffn_activation='relu', kernel_regularizer=None, block_index=0): # Added kernel_regularizer
    # Pre-LN structure
    # Attention Block
    normed_inputs_for_mha = LayerNormalization(epsilon=1e-6, name=f"ln_mha_pre_block{block_index}")(inputs) # Unique name
    attention_layer = MultiHeadAttention( # Instantiate the layer first
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout, # This is MHA's internal attention dropout
        kernel_regularizer=kernel_regularizer, # Apply L2 to MHA's internal dense layers
        name=f"mha_block{block_index}" # Added unique name to MHA
    )
    # Call the layer, passing use_causal_mask here
    attention_output = attention_layer(
        normed_inputs_for_mha, 
        normed_inputs_for_mha, 
        use_causal_mask=True # Pass here for causality
    ) 
    attention_output = Dropout(dropout)(attention_output)
    # Add MHA output back to the *original unnormalized* inputs
    residual_after_mha = Add(name=f"add_mha_residual_block{block_index}")([attention_output, inputs]) # Unique name

    # Feed Forward Block
    normed_inputs_for_ffn = LayerNormalization(epsilon=1e-6, name=f"ln_ffn_pre_block{block_index}")(residual_after_mha) # Unique name
    # FFN operates on normalized inputs from the MHA block's output
    ffn_output = Dense(ff_dim, activation=ffn_activation, kernel_regularizer=kernel_regularizer, name=f"ffn_dense_1_block{block_index}")(normed_inputs_for_ffn) # Added L2, Unique name
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(inputs.shape[-1], kernel_regularizer=kernel_regularizer, name=f"ffn_dense_2_block{block_index}")(ffn_output) # Added L2, Project back, unique name
    # Add FFN output back to the output of the MHA block (residual_after_mha)
    output = Add(name=f"add_ffn_residual_block{block_index}")([ffn_output, residual_after_mha]) # Unique name
    
    return output

# +++ Positional Encoding Layer +++
# ... (Keep implementation) ...
class PositionalEncoding(Layer):
    # ... (Implementation) ...
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for standard positional encoding, got {d_model}")
    def build(self, input_shape):
        super().build(input_shape)
        pos = np.arange(self.max_len)[:, np.newaxis]
        i = np.arange(self.d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config

# +++ NEW Custom Attention Pooling Layer +++
class AttentionPooling(Layer):
    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        # No specific tunable parameters needed for a simple version from Optuna for now
        # The main learnable weight will be the context vector, initialized in build()

    def build(self, input_shape):
        # input_shape expected to be (batch_size, sequence_length, num_features)
        # num_features is d_model
        self.d_model = input_shape[-1]
        
        # Learnable context vector (the "query" or "learned question")
        # This weight will be learned during training.
        self.context_vector = self.add_weight(
            name='attention_context_vector',
            shape=(self.d_model, 1), # Shape for dot product with each time step's features
            initializer='glorot_uniform', # A common initializer
            trainable=True
        )
        super(AttentionPooling, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length, d_model)
        
        # Calculate alignment scores (dot product of each time step with the context vector)
        # einsum is a concise way to do a batched dot product:
        # 'bsf,fo->bs' means: batch, sequence, features_input dot features_context,output_channels -> batch,sequence
        # In our case, context_vector is (d_model, 1), so f=d_model, o=1.
        # The result of K.squeeze will be (batch_size, sequence_length)
        attention_scores = K.squeeze(K.dot(inputs, self.context_vector), axis=-1)
        # attention_scores shape: (batch_size, sequence_length)

        # Apply softmax to get attention weights
        # Softmax is applied along the sequence_length dimension
        attention_weights = Softmax(axis=1, name="attention_weights")(attention_scores)
        # attention_weights shape: (batch_size, sequence_length)

        # Expand dims of attention_weights to be (batch_size, sequence_length, 1)
        # so it can be broadcasted for element-wise multiplication with inputs
        attention_weights_expanded = K.expand_dims(attention_weights, axis=-1)
        
        # Compute the weighted sum of the input sequence
        # inputs shape: (batch_size, sequence_length, d_model)
        # attention_weights_expanded shape: (batch_size, sequence_length, 1)
        # Result after multiply: (batch_size, sequence_length, d_model)
        # Result after K.sum along sequence_length axis: (batch_size, d_model)
        weighted_sum = K.sum(inputs * attention_weights_expanded, axis=1)
        
        return weighted_sum # Shape: (batch_size, d_model)

    def compute_output_shape(self, input_shape):
        # Output shape will be (batch_size, d_model)
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        # Store any fixed configuration parameters if there were any
        config = super(AttentionPooling, self).get_config()
        # No extra config params for this simple version.
        # If we added tunable dropout or other things to this layer, they'd go here.
        return config
# +++ End NEW Custom Attention Pooling Layer +++

# +++ NEW RevIN Layer (Instance Normalization for Time Series) +++
class RevIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, affine=True, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.affine = affine
        self.supports_masking = True # Important if there's masking upstream

    def build(self, input_shape): # Expects (batch, seq_len, features)
        if len(input_shape) != 3:
            raise ValueError(f"RevIN layer expects a 3D input (batch, sequence_length, features), but got shape: {input_shape}")
        
        self.num_features = input_shape[-1]
        if self.affine:
            self.gamma = self.add_weight(name='revin_gamma',
                                         shape=(1, 1, self.num_features), # Shape for broadcasting
                                         initializer='ones',
                                         trainable=True)
            self.beta = self.add_weight(name='revin_beta',
                                        shape=(1, 1, self.num_features), # Shape for broadcasting
                                        initializer='zeros',
                                        trainable=True)
        super().build(input_shape)

    def call(self, x, training=None): # x shape: (batch, seq_len, features)
        # Calculate mean and variance per feature channel, per instance, across sequence length
        # K.mean(x, axis=1, keepdims=True) will give shape (batch, 1, features)
        # These statistics are instance-specific and should not leak information across the batch during training.
        # Using stop_gradient as these are treated as per-instance statistics for normalization,
        # not parameters to be learned directly through backpropping into their calculation.
        instance_mean = tf.stop_gradient(K.mean(x, axis=1, keepdims=True))
        instance_variance = tf.stop_gradient(K.var(x, axis=1, keepdims=True))
        
        normalized = (x - instance_mean) / K.sqrt(instance_variance + self.epsilon)
        
        if self.affine:
            normalized = self.gamma * normalized + self.beta
            
        return normalized 

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon, 'affine': self.affine})
        return config
# +++ END NEW RevIN Layer +++

# +++ NEW Single-Stack CNN-Transformer Model Definition +++
def build_model(
    sequence_length,
    n_features_total, # Original number of features before embedding
    all_feature_names, # List of all feature names in order
    categorical_feature_specs, # Dict: {'feat_name': {'input_dim': N, 'output_dim': M}, ...}
    # --- Instance Norm specific args ---
    use_instance_norm,      # bool
    instance_norm_affine, # bool
    # --- CNN specific args ---
    cnn_layer_configs, 
    cnn_pool_size,
    cnn_pool_strides,
    cnn_dropout_rate,
    cnn_activation,
    cnn_use_bias,
    num_transformer_blocks,
    num_heads,
    head_size,          
    ff_dim,             
    mlp_units,          
    transformer_dropout_rate,
    mlp_dropout_rate,
    l2_strength,
    ffn_activation, # Added ffn_activation
    output_bias_init_value=None # Default argument should be last among these
):
    """
    Builds a single-stack CNN-Transformer model with Optuna-tuned hyperparameters,
    now with support for embedding categorical features.
    """
    logger.info("Building single-stack CNN-Transformer model with dynamic Optuna HPs & Embedding Layers...")
    
    kernel_reg = regularizers.l2(l2_strength) if l2_strength > 0 else None

    inputs = Input(shape=(sequence_length, n_features_total), name="input_features_raw")
    
    # --- 0. Feature Processing: Embedding for Categoricals, Passthrough for Numerical ---
    logger.info("Processing input features: Applying embeddings to categoricals...")
    numerical_feature_indices = []
    
    # Identify indices of numerical and categorical features based on all_feature_names
    for i, name in enumerate(all_feature_names):
        if name not in categorical_feature_specs:
            numerical_feature_indices.append(i)
            
    processed_feature_streams = []

    # Process Categorical Features with Embedding Layers
    for cat_feature_name, spec in categorical_feature_specs.items():
        if cat_feature_name in all_feature_names:
            cat_feature_index = all_feature_names.index(cat_feature_name)
            
            # Slice out this single categorical feature column
            # Input shape for Lambda: (batch, sequence_length, n_features_total)
            # Output shape of Lambda: (batch, sequence_length, 1)
            cat_input_stream = Lambda(
                lambda z, index=cat_feature_index: z[:, :, index:index+1], 
                name=f"slice_{cat_feature_name}"
            )(inputs)
            
            # Squeeze the last dimension to make it (batch, sequence_length) for Embedding layer
            cat_input_squeezed = Lambda(
                lambda z: K.squeeze(z, axis=-1),
                name=f"squeeze_{cat_feature_name}"
            )(cat_input_stream)

            embedding_layer = Embedding(
                input_dim=spec['input_dim'],         # Vocabulary size
                output_dim=spec['output_dim'],       # Embedding vector size
                input_length=sequence_length,        # Max length of sequence
                name=f"embed_{cat_feature_name}"
            )(cat_input_squeezed)
            # Output of Embedding: (batch, sequence_length, spec['output_dim'])
            processed_feature_streams.append(embedding_layer)
            logger.info(f"  Embedded {cat_feature_name} -> shape: {embedding_layer.shape}")
        else:
            logger.warning(f"  Categorical feature '{cat_feature_name}' defined in specs but not found in 'all_feature_names'. Skipping embedding.")

    # Process Numerical Features (gather them)
    if numerical_feature_indices:
        # Slice out all numerical features. Using a list of Lambda layers and then concatenating.
        numerical_slices = []
        for num_idx in numerical_feature_indices:
            num_feat_name = all_feature_names[num_idx]
            num_slice = Lambda(
                lambda z, index=num_idx: z[:, :, index:index+1],
                name=f"slice_numerical_{num_feat_name.replace('-', '_')}" # Ensure valid layer name
            )(inputs)
            numerical_slices.append(num_slice)
        
        if len(numerical_slices) > 1:
            numerical_block = Concatenate(name="numerical_features_block")(numerical_slices)
        elif len(numerical_slices) == 1:
            numerical_block = numerical_slices[0] # Avoid unnecessary Concat
        else: # Should not happen if numerical_feature_indices is not empty
            numerical_block = None 
            
        if numerical_block is not None:
            processed_feature_streams.append(numerical_block)
            logger.info(f"  Numerical features block shape: {numerical_block.shape}")
        else:
            logger.info("  No numerical features to process as a block.")
            
    else:
        logger.info("  No numerical features identified.")

    # Concatenate all processed streams (embeddings + numerical block)
    if len(processed_feature_streams) > 1:
        x = Concatenate(name="features_concatenated")(processed_feature_streams)
    elif len(processed_feature_streams) == 1:
        x = processed_feature_streams[0]
    else:
        logger.error("CRITICAL: No feature streams (neither categorical nor numerical) were processed. Model input is empty.")
        # This is a fatal error for model building.
        raise ValueError("No feature streams available after processing. Check feature lists and specs.")
    
    logger.info(f"Final concatenated feature tensor shape (input to potential RevIN): {x.shape}")

    # --- NEW: Optional RevIN Layer ---
    if use_instance_norm:
        logger.info(f"Applying RevIN layer (Instance Normalization). Affine: {instance_norm_affine}")
        x = RevIN(affine=instance_norm_affine, name="revin_layer")(x)
        logger.info(f"Shape after RevIN layer: {x.shape}")
    # --- End RevIN Layer ---
    
    logger.info(f"Final concatenated feature tensor shape (input to CNNs): {x.shape}")
    
    # --- 1. CNN Stack ---
    logger.info("Building CNN stack with Gated Linear Units (GLUs)...") # Updated log
    current_seq_len = sequence_length
    for i, layer_conf in enumerate(cnn_layer_configs):
        residual = x # Store input for residual connection

        # GLU Implementation
        # Data Path Convolution
        data_path_conv = Conv1D(
            filters=layer_conf['filters'],
            kernel_size=layer_conf['kernel'],
            dilation_rate=layer_conf['dilation'],
            activation=cnn_activation, # Main activation for the data path
            use_bias=cnn_use_bias,
            padding='causal', 
            kernel_regularizer=kernel_reg,
            name=f"cnn_glu_data_path_conv_{i+1}"
        )(x) # Input to GLU block is x (which is residual)

        # Gate Path Convolution
        gate_path_conv = Conv1D(
            filters=layer_conf['filters'], # Gate path must have same number of filters to multiply
            kernel_size=layer_conf['kernel'],
            dilation_rate=layer_conf['dilation'],
            activation='sigmoid',      # Sigmoid activation for the gate
            use_bias=cnn_use_bias,
            padding='causal', 
            kernel_regularizer=kernel_reg,
            name=f"cnn_glu_gate_path_conv_{i+1}"
        )(x) # Input to GLU block is x (which is residual)

        # Gating Operation
        gated_activation_output = multiply([data_path_conv, gate_path_conv], name=f"cnn_glu_multiply_{i+1}")
        
        # Original CNN layers after the activation (now after GLU)
        # The 'x' here becomes the output of the GLU block
        x_after_glu = gated_activation_output 
        
        x_after_glu = BatchNormalization(name=f"cnn_bn_{i+1}")(x_after_glu)
        if cnn_dropout_rate > 0:
            x_after_glu = SpatialDropout1D(cnn_dropout_rate, name=f"cnn_spatial_dropout_{i+1}")(x_after_glu) 
        
        # Add residual connection
        # The residual 'residual' (input to the GLU block) is added to 'x_after_glu'
        if residual.shape[-1] != x_after_glu.shape[-1]:
            # This case should ideally not happen if layer_conf['filters'] is consistent
            # and matches the expected output dimension of the GLU.
            # However, if residual comes from a different feature dimension than GLU output, projection is needed.
            # For GLU, layer_conf['filters'] defines the output channels of the multiply operation.
            projected_residual = Conv1D(
                filters=x_after_glu.shape[-1], 
                kernel_size=1,      
                padding='causal',   
                kernel_regularizer=kernel_reg,
                name=f"cnn_residual_projection_{i+1}"
            )(residual)
            logger.info(f"  CNN Layer {i+1} residual projected. Shape: {projected_residual.shape}")
        else:
            projected_residual = residual

        x = Add(name=f"cnn_residual_add_{i+1}")([x_after_glu, projected_residual])
        logger.info(f"  CNN Layer {i+1} (GLU): filters={layer_conf['filters']}, kernel={layer_conf['kernel']}, dilation={layer_conf['dilation']}. Shape after GLU/BN/dropout & residual add: {x.shape}")

    # Store the number of features output by the main CNN stack
    cnn_output_features = x.shape[-1]

    # --- Replace MaxPooling1D with Strided Conv1D for learned downsampling ---
    if cnn_pool_size and cnn_pool_strides and cnn_pool_size > 1 and cnn_pool_strides > 0 : # cnn_pool_size check might be less relevant, strides is key
        x = Conv1D(
            filters=cnn_output_features, # Maintain the feature dimension
            kernel_size=2, # Kernel size for the strided convolution, 2 for a direct conceptual replacement of pool_size=2
            strides=cnn_pool_strides, # This will be 2
            padding='same', # Ensures output sequence length is ceil(input_length / strides)
            activation=cnn_activation, # Use the same activation as other CNNs
            use_bias=cnn_use_bias,    # Use from args
            kernel_regularizer=kernel_reg,
            name="strided_conv_downsample"
        )(x)
        if current_seq_len is not None: # current_seq_len should be an int here
             current_seq_len = (current_seq_len + cnn_pool_strides - 1) // cnn_pool_strides
        logger.info(f"  Strided Conv1D Downsampling: filters={cnn_output_features}, kernel=2, strides={cnn_pool_strides}. Shape after: {x.shape}. New est. seq_len: {current_seq_len}")
    # Original MaxPooling1D (commented out or removed)
    # if cnn_pool_size and cnn_pool_strides and cnn_pool_size > 1 and cnn_pool_strides > 0 : # Ensure pool_size > 1
    #     x = MaxPooling1D(
    #         pool_size=cnn_pool_size,
    #         strides=cnn_pool_strides,
    #         padding='same', 
    #         name="cnn_maxpool"
    #     )(x)
    #     # Update sequence length after pooling
    #     # With 'same' padding and strides S, new length is ceil(L_in / S)
    #     if current_seq_len is not None: # current_seq_len should be an int here
    #          current_seq_len = (current_seq_len + cnn_pool_strides - 1) // cnn_pool_strides
    #     logger.info(f"  CNN MaxPooling: pool_size={cnn_pool_size}, strides={cnn_pool_strides}. Shape after: {x.shape}. New est. seq_len: {current_seq_len}")
    
    # +++ NEW: Feature Gating Mechanism +++
    # The input 'x' here is (batch_size, sequence_length_after_pooling, cnn_output_features)
    # We want to learn a gate for each feature channel.
    feature_gates = Dense(
        units=cnn_output_features, # Output one gate value per feature channel
        activation='sigmoid',      # Sigmoid to keep gates between 0 and 1
        kernel_regularizer=kernel_reg, 
        name='feature_gate_dense'
    )(x) # Apply to the full sequence tensor x, dense layer operates on the last dim
    
    x = multiply([x, feature_gates], name='gated_features') # Element-wise multiplication
    logger.info(f"Shape after Feature Gating (gates applied): {x.shape}")
    # +++ END: Feature Gating Mechanism +++
    
    # cnn_output_features = x.shape[-1] # This line might be redundant if cnn_output_features is defined before strided conv and strided conv maintains it.
    # Use Keras symbolic shape for pooled_sequence_length if available, otherwise use calculated current_seq_len
    pooled_sequence_length = x.shape[1] if x.shape[1] is not None else current_seq_len

    if pooled_sequence_length is None: 
        raise ValueError("Sequence length became None after CNN stack / pooling.")
    logger.info(f"  CNN stack output shape: {x.shape}. Features: {cnn_output_features}, SeqLen for PE: {pooled_sequence_length}")

    # --- 2. Prepare for Transformer (Projection if needed) ---
    d_model_transformer = num_heads * head_size 
    logger.info(f"Target Transformer d_model: {d_model_transformer} (heads={num_heads}, head_size={head_size})")

    if cnn_output_features != d_model_transformer:
        logger.info(f"Projecting CNN output features from {cnn_output_features} to {d_model_transformer} for Transformer input.")
        x = Dense(d_model_transformer, kernel_regularizer=kernel_reg, name="cnn_to_transformer_projection")(x)
        logger.info(f"Shape after projection: {x.shape}")
    
    # --- 3. Positional Encoding ---
    # d_model_transformer should be even due to num_heads * head_size (all head_size options are even)
    if d_model_transformer % 2 != 0:
         logger.error(f"CRITICAL: Transformer d_model ({d_model_transformer}) is ODD. This will cause PositionalEncoding to fail. Check head_size/num_heads Optuna choices.")
         # This case should ideally not be reached given Optuna's categorical choices for head_size.
         # If it is, PositionalEncoding will raise a ValueError.

    x = PositionalEncoding(max_len=pooled_sequence_length, d_model=d_model_transformer, name="positional_encoding")(x)
    logger.info(f"Shape after Positional Encoding: {x.shape}")

    # --- 4. Transformer Blocks ---
    logger.info(f"Building {num_transformer_blocks} Transformer blocks...")
    for i in range(num_transformer_blocks):
        # ff_dim is now correctly passed as the full dimension for the feed-forward sub-layer
        x = transformer_encoder( 
            inputs=x, 
            head_size=head_size, # This is key_dim for MHA
            num_heads=num_heads, 
            ff_dim=ff_dim, # Tuned ff_dim (calculated in objective as d_model_transformer_mha * ff_dim_factor)
            dropout=transformer_dropout_rate, # Use tuned transformer dropout
            ffn_activation=ffn_activation, # Pass ffn_activation
            kernel_regularizer=kernel_reg, # Pass the L2 regularizer
            block_index=i # Pass block index for unique layer names
        )
        logger.info(f"  Shape after Transformer block {i+1}: {K.int_shape(x)}")

    # --- 5. Aggregate Transformer Output ---
    # x = GlobalAveragePooling1D(name="transformer_global_avg_pool")(x) # OLD: Replaced with AttentionPooling
    x = AttentionPooling(name="transformer_attention_pooling")(x) # NEW: Using custom AttentionPooling
    logger.info(f"Shape after Attention Pooling: {x.shape}")

    # --- 6. MLP Head (Classifier) ---
    logger.info("Building MLP head...")
    for i, units in enumerate(mlp_units): # mlp_units is a list of unit sizes
        x = Dense(units, activation="gelu", kernel_regularizer=kernel_reg, name=f"mlp_dense_{i+1}")(x) # Changed to gelu
        if mlp_dropout_rate > 0: # Use tuned MLP dropout
            x = Dropout(mlp_dropout_rate, name=f"mlp_dropout_{i+1}")(x)
        logger.info(f"  MLP Layer {i+1}: units={units}. Shape after: {x.shape}")
    
    # --- 7. Final Output ---
    output_bias = tf.keras.initializers.Constant(output_bias_init_value) if output_bias_init_value is not None else None
    outputs = Dense(1, activation="sigmoid", kernel_regularizer=kernel_reg, name="output_layer", bias_initializer=output_bias)(x) # Sigmoid for binary classification
    logger.info(f"Output layer shape: {outputs.shape}. Bias init: {'Constant' if output_bias else 'Default'}")

    model = Model(inputs=inputs, outputs=outputs, name="chimera_single_stack_cnn_transformer")
    logger.info("Single-stack CNN-Transformer model built successfully.")

    return model
# --- End NEW Single-Stack CNN-Transformer Model Definition ---


# --- OLD SINGLE INPUT MODEL FUNCTION - RENAME/REMOVE LATER --- 
# def build_cnn_transformer_model(...):
#    ...
# --- END OLD MODEL FUNCTION ---


# +++ Custom Callback for Manual Progressive History Saving +++
# ... (Keep implementation) ...
class ManualHistorySaver(tf.keras.callbacks.Callback):
    # ... (Implementation) ...
    def __init__(self, filepath):
        super(ManualHistorySaver, self).__init__()
        self.filepath = Path(filepath)
        self.epoch_history = {}
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    def on_train_begin(self, logs=None):
        self.epoch_history = {}
        try:
            with open(self.filepath, 'w') as f: json.dump({}, f, indent=4)
            logger.info(f"ManualHistorySaver: Initialized history file at {self.filepath}")
        except Exception as e:
            logger.error(f"ManualHistorySaver: Error initializing history file {self.filepath}: {e}", exc_info=True)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(f"ManualHistorySaver: Reached end of epoch {epoch+1}. Processing logs.")
        logger.debug(f"ManualHistorySaver: Logs received for epoch {epoch+1}: {logs}")
        for k, v in logs.items():
            log_value = v
            if isinstance(v, np.generic): log_value = v.item()
            try:
                 serializable_value = float(log_value)
                 self.epoch_history.setdefault(k, []).append(serializable_value)
            except (TypeError, ValueError) as e:
                 logger.warning(f"ManualHistorySaver: Could not convert log value for key '{k}' (value: {log_value}, type: {type(log_value)}) to float. Skipping key for epoch {epoch+1}. Error: {e}")
        if self.epoch_history:
            try:
                logger.info(f"ManualHistorySaver: Attempting to save accumulated history to {self.filepath}")
                with open(self.filepath, 'w') as f: json.dump(self.epoch_history, f, indent=4)
                logger.info(f"ManualHistorySaver: Successfully saved history up to epoch {epoch+1} to {self.filepath}")
            except Exception as e:
                logger.error(f"ManualHistorySaver: Error saving accumulated history to {self.filepath}: {e}", exc_info=True)
        else:
             logger.warning(f"ManualHistorySaver: Accumulated history is empty after epoch {epoch+1}. Not saving.")

# +++ F1EvalCallback Definition (copied from o3_responses_4.md, with minor adjustments) +++
class F1EvalCallback(tf.keras.callbacks.Callback):
    """
    Callback to evaluate F1 score on validation data at the end of each epoch
    and log the best F1 score and corresponding threshold found so far.
    Uses a tf.data.Dataset for validation.
    """
    def __init__(self, val_dataset, history_f1_path, patience=0, num_val_samples=None, val_steps=None): # Added val_steps
        super().__init__()
        self.val_dataset = val_dataset # Store the tf.data.Dataset
        self.num_val_samples = num_val_samples # Approx number of samples for progress bar in predict
        self.val_steps = val_steps # Number of steps to iterate for one pass over val_dataset

        self.history_f1_path = history_f1_path
        try:
            os.makedirs(os.path.dirname(self.history_f1_path), exist_ok=True)
        except Exception as e_mkdir:
            logger.error(f"F1EvalCallback: Error creating directory for history_f1_path {self.history_f1_path}: {e_mkdir}")
        if os.path.exists(self.history_f1_path):
            try:
                os.remove(self.history_f1_path)
                logger.info(f"F1EvalCallback: Removed existing F1 history file: {self.history_f1_path}")
            except Exception as e_rm:
                logger.error(f"F1EvalCallback: Error removing existing F1 history file {self.history_f1_path}: {e_rm}")
        logger.info(f"F1EvalCallback initialized. Results will be saved to {self.history_f1_path}")

        self.best_f1 = -np.inf
        self.best_threshold = 0.5
        self.best_epoch = -1
        self.current_epoch_val_f1 = 0.0
        self.current_epoch_thr = 0.5

        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        try:
            if self.val_dataset is None:
                logger.warning(f"F1EvalCallback: val_dataset is None. Skipping F1 calculation for epoch {epoch+1}.")
                self._set_default_logs(logs)
                return

            # Get predictions
            # Ensure val_dataset doesn't repeat if it has .repeat() for model.fit.
            # model.predict should handle this if steps are provided or dataset is finite.
            logger.debug(f"F1EvalCallback: Predicting on val_dataset with {self.val_steps} steps.")
            y_prob = self.model.predict(self.val_dataset, steps=self.val_steps, verbose=0).flatten() # Added steps=self.val_steps
            logger.debug(f"F1EvalCallback: Prediction complete. y_prob shape: {y_prob.shape}")

            # Get true labels by iterating through the dataset
            y_true_list = []
            # Iterate for one pass over the validation data
            dataset_to_iterate = self.val_dataset
            if self.val_steps is not None and self.val_steps > 0:
                dataset_to_iterate = self.val_dataset.take(self.val_steps)
                logger.debug(f"F1EvalCallback: Taking {self.val_steps} steps from val_dataset to get y_true.")
            else:
                logger.warning(f"F1EvalCallback: val_steps not provided or invalid ({self.val_steps}). Iterating val_dataset directly for y_true. This might loop if dataset repeats.")

            for _, labels_batch in dataset_to_iterate: 
                y_true_list.append(labels_batch.numpy())
            
            if not y_true_list:
                logger.warning(f"F1EvalCallback: Extracted no true labels from val_dataset. Skipping F1 calculation.")
                self._set_default_logs(logs)
                return

            y_val_true_np = np.concatenate(y_true_list).flatten()
            
            # If y_prob is longer than y_val_true_np due to dataset repeating and predict not having steps, truncate y_prob.
            # This can happen if val_dataset used in model.fit has .repeat() and model.predict also sees that.
            if len(y_prob) > len(y_val_true_np) and self.num_val_samples is not None and len(y_val_true_np) == self.num_val_samples:
                logger.warning(f"F1EvalCallback: y_prob length ({len(y_prob)}) > y_val_true_np length ({len(y_val_true_np)}). Truncating y_prob to {len(y_val_true_np)}.")
                y_prob = y_prob[:len(y_val_true_np)]
            elif len(y_prob) < len(y_val_true_np) and self.num_val_samples is not None and len(y_val_true_np) == self.num_val_samples:
                 logger.warning(f"F1EvalCallback: y_prob length ({len(y_prob)}) < y_val_true_np length ({len(y_val_true_np)}). This might be an issue. len(y_val_true_np) was taken from num_val_samples: {self.num_val_samples}")
                 # Potentially pad y_prob or take subset of y_val_true_np. For now, this is a warning.
                 # This case should be less common if predict iterates fully.

            if len(y_prob) != len(y_val_true_np):
                logger.error(f"F1EvalCallback: y_prob length ({len(y_prob)}) and y_val_true_np length ({len(y_val_true_np)}) mismatch AFTER potential truncation. Skipping F1 calculation. num_val_samples: {self.num_val_samples}")
                self._set_default_logs(logs)
                return
            
            if len(y_val_true_np) == 0:
                logger.warning(f"F1EvalCallback: y_val_true_np is empty after concatenation. Skipping F1 calculation.")
                self._set_default_logs(logs)
                return

            best_f1_epoch, best_thr_epoch = 0, 0.5
            thresholds = np.linspace(0.01, 0.99, 99)
            f1_scores = [f1_score(y_val_true_np, (y_prob >= t).astype(int)) for t in thresholds] # Use y_val_true_np
            best_f1_epoch = np.max(f1_scores)
            best_thr_epoch = thresholds[np.argmax(f1_scores)]

            logger.info(f" - val_f1_opt: {best_f1_epoch:.4f} (thr={best_thr_epoch:.2f})")

            if best_f1_epoch > self.best_f1:
                self.best_f1 = best_f1_epoch
                self.best_threshold = best_thr_epoch
                self.best_epoch = epoch + 1
                self.wait = 0
                logger.info(f"F1EvalCallback: New best val_f1: {self.best_f1:.4f} at epoch {self.best_epoch} (thr={self.best_threshold:.2f})")
            else:
                self.wait += 1

            logs['best_val_f1'] = self.best_f1
            logs['best_thr'] = self.best_threshold
            logs['current_epoch_val_f1'] = best_f1_epoch
            logs['current_epoch_thr'] = best_thr_epoch
            self.current_epoch_val_f1 = best_f1_epoch
            self.current_epoch_thr = best_thr_epoch

            try:
                with open(self.history_f1_path, "a") as f:
                    json.dump({"epoch": epoch + 1,
                               "current_val_f1": float(best_f1_epoch),
                               "current_thr": float(best_thr_epoch),
                               "best_val_f1": float(self.best_f1),
                               "best_thr": float(self.best_threshold)}, f)
                    f.write("\n") # Moved inside the 'with' block
            except Exception as e_write:
                logger.error(f"F1EvalCallback: Failed to write F1 history to {self.history_f1_path}: {e_write}")

            if self.patience > 0 and self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                logger.warning(f"F1EvalCallback: Stopping training early at epoch {epoch + 1} as val_f1 did not improve for {self.patience} epochs.")

        except Exception as e_callback:
            logger.error(f"F1EvalCallback: Error during on_epoch_end for epoch {epoch+1}: {e_callback}")
            import traceback
            logger.error(traceback.format_exc())
            self._set_default_logs(logs)
            
    def _set_default_logs(self, logs):
            logs['best_val_f1'] = logs.get('best_val_f1', self.best_f1 if hasattr(self, 'best_f1') else 0.0)
            logs['best_thr'] = logs.get('best_thr', self.best_threshold if hasattr(self, 'best_threshold') else 0.5)
            logs['current_epoch_val_f1'] = 0.0
            logs['current_epoch_thr'] = 0.5
            self.current_epoch_val_f1 = 0.0
            self.current_epoch_thr = 0.5

# +++ Custom Callback to Stop on Consistently Low val_recall +++
class ValRecallStopper(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_recall', target_recall=0.0, patience=5, verbose=1):
        super(ValRecallStopper, self).__init__()
        self.monitor = monitor
        self.target_recall = target_recall
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        logger.info(f"ValRecallStopper: Monitoring '{self.monitor}' to be > {self.target_recall} with patience {self.patience}.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_val_recall = logs.get(self.monitor)

        if current_val_recall is None:
            if self.verbose > 0:
                logger.warning(f"ValRecallStopper: Metric '{self.monitor}' not found in logs. Skipping check for epoch {epoch + 1}.")
            return

        if current_val_recall <= self.target_recall:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose > 0:
                    logger.info(f"Epoch {epoch + 1}: ValRecallStopper stopping training. '{self.monitor}' ({current_val_recall:.4f}) was <= {self.target_recall} for {self.wait} consecutive epochs (patience {self.patience}).")
        else:
            self.wait = 0 # Reset counter if val_recall is above target

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            logger.info(f"ValRecallStopper: Training stopped early at epoch {self.stopped_epoch + 1}.")

class LinearWarmUp(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs, base_lr, run_dir): # Added run_dir for logging
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.run_dir = run_dir # Store run_dir
        # Ensure run_dir is a Path object for consistency, then join
        self.lr_schedule_log_path = Path(self.run_dir) / "warmup_lr.txt"
        self.lr_history = [] # To store (epoch, lr)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            k = (epoch + 1) / self.warmup_epochs  # 0 -> 1
            new_lr = self.base_lr * k
            self.model.optimizer.learning_rate.assign(new_lr) # Use .learning_rate.assign()
            current_lr = self.model.optimizer.learning_rate.numpy() # Use .learning_rate.numpy()
            self.lr_history.append((epoch + 1, float(current_lr))) # Log epoch (1-indexed) and LR
            # print(f"Epoch {epoch+1}/{self.warmup_epochs}: Warm-up LR set to {current_lr:.8f}") # Optional: for dry run

    def on_train_end(self, logs=None): # Save LRs at the end of training
        if self.lr_history:
            try:
                # Ensure the directory exists
                Path(self.run_dir).mkdir(parents=True, exist_ok=True)
                with open(self.lr_schedule_log_path, 'w') as f:
                    f.write("Epoch,LearningRate\\n")
                    for epoch_num, lr_val in self.lr_history:
                        f.write(f"{epoch_num},{lr_val:.8f}\\n")
                logger.info(f"Warm-up LR schedule saved to {self.lr_schedule_log_path}")
            except Exception as e:
                logger.error(f"LinearWarmUp: Error saving LR schedule to {self.lr_schedule_log_path}: {e}")

    def get_config(self): # For Keras serialization, if ever needed
        config = super().get_config()
        config.update({
            'warmup_epochs': self.warmup_epochs,
            'base_lr': self.base_lr,
            'run_dir': str(self.run_dir) # Ensure run_dir is string for serialization
        })
        return config

# --- Callbacks ---
# REVISED for Multi-Branch: Pass val_inputs list to F1EvalCallback
def get_callbacks(args, run_dir, val_dataset, batch_size, hp, num_val_sequences_possible=None, val_steps_for_callback=None): # Added val_steps_for_callback
    callbacks = []

    # Checkpoint saver for best model based on val_loss (standard Keras callback)
    if args.save_results: # Only add checkpoints if save_results is True
        # Filename as per spec: best_auc_model.keras (within the trial-specific run_dir_path)
        checkpoint_auc_path = run_dir / 'best_auc_model.keras' 
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_auc_path,
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ))
        logger.info(f"ModelCheckpoint (val_auc) enabled, saving to {checkpoint_auc_path}")

    # ReduceLROnPlateau patience from Phase 2 spec: 2
    # ReduceLROnPlateau factor from Phase 2 spec: 0.5
    logger.info("ReduceLROnPlateau callback REMOVED in favor of CosineDecay schedule with Warmup.")

    # EarlyStopping
    # EarlyStopping patience from Phase 2 spec: 5
    callbacks.append(EarlyStopping(
        monitor='val_f1_score', # MODIFIED: Monitor val_f1_score from F1EvalCallback
        patience=args.early_stopping_patience, # Use CLI arg (default 4)
        verbose=1,
        mode='max',
        restore_best_weights=True # As per Phase 2 spec
    ))
    logger.info(f"EarlyStopping enabled: monitor='val_f1_score', patience={args.early_stopping_patience}, restore_best_weights=True")
    
    # --- ValRecallStopper (o3 aggressive early stop) ---
    # Stop if val_recall doesn't reach target_recall (e.g., 0.1) for 'patience' epochs
    # val_recall_stopper = ValRecallStopper(
    #     monitor='val_recall', 
    #     target_recall=0.1, # o3: e.g. 0.1
    #     patience=3,        # o3: e.g. 2 epochs. MODIFIED to 3 based on user feedback.
    #     verbose=1
    # )
    # callbacks.append(val_recall_stopper)
    # logger.info(f"ValRecallStopper enabled: monitor='val_recall', target_recall=0.1, patience=3 (o3 aggressive early stop, modified patience)") # Updated log
    logger.info("ValRecallStopper is explicitly DISABLED for Optuna runs to rely on val_auc EarlyStopping.")

    # 5. TensorBoard, ManualHistorySaver (optional, kept if args.save_results)
    if args.save_results: 
        tensorboard_log_dir = run_dir / "logs" / f"trial_{hp.get('trial_number_for_paths', 'XXX')}" / "fit" / datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1))
        logger.info(f"TensorBoard logging enabled to {tensorboard_log_dir}")

        history_json_path = run_dir / f'history_{hp.get('trial_number_for_paths', 'XXX')}.json'
        callbacks.append(ManualHistorySaver(filepath=history_json_path))
        logger.info(f"ManualHistorySaver enabled, saving history to {history_json_path}")

    # F1 Evaluation Callback
    f1_history_path = run_dir / f"f1_history_trial_{hp.get('trial_number_for_paths', 'XXX')}.jsonl"
    f1_eval_callback = F1EvalCallback(
        val_dataset=val_dataset, 
        history_f1_path=f1_history_path,
        patience=args.f1_callback_patience,
        num_val_samples=num_val_sequences_possible,
        val_steps=val_steps_for_callback # Pass val_steps_for_callback
    )
    callbacks.append(f1_eval_callback)

    return callbacks

# +++ Helper: Scale Features +++
def scale_features(X_train_flat_df, X_val_flat_df, X_test_flat_df, normalization_method, save_path=None, trial_number=None):
    """
    Scales the flat feature DataFrames.

    Args:
        X_train_flat_df (pd.DataFrame): Training features.
        X_val_flat_df (pd.DataFrame): Validation features.
        X_test_flat_df (pd.DataFrame): Test features.
        normalization_method (str): 'standard', 'minmax', 'robust', 'power_yeo_johnson', 
                                    'quantile_normal', 'quantile_uniform', or 'none'.
        save_path (Path, optional): Path to save the fitted scaler. Defaults to None.
        trial_number (int, optional): Optuna trial number for random_state in QuantileTransformer.

    Returns:
        tuple: (X_train_scaled, X_val_scaled, X_test_scaled, scaler_object)
               Returns original values and None for scaler if normalization_method is 'none'.
    """
    X_train_flat_scaled = X_train_flat_df.values
    X_val_flat_scaled = X_val_flat_df.values
    X_test_flat_scaled = X_test_flat_df.values
    scaler = None

    if normalization_method != 'none':
        logger.info(f"Normalizing flat features using '{normalization_method}' scaler...")
        if normalization_method == 'standard':
            scaler = StandardScaler()
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif normalization_method == 'robust':
            scaler = RobustScaler()
        elif normalization_method == 'power_yeo_johnson':
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        elif normalization_method == 'quantile_normal':
            # Use trial_number for random_state if QuantileTransformer supports it and it's beneficial for reproducibility
            # For scikit-learn >= 0.24, random_state is available.
            # Assuming it's available. If not, this might need adjustment or version check.
            scaler = QuantileTransformer(output_distribution='normal', random_state=trial_number if trial_number is not None else 42)
        elif normalization_method == 'quantile_uniform':
            scaler = QuantileTransformer(output_distribution='uniform', random_state=trial_number if trial_number is not None else 42)
        else:
            logger.error(f"Invalid normalization method: {normalization_method} in scale_features. Returning unscaled data.")
            return X_train_flat_scaled, X_val_flat_scaled, X_test_flat_scaled, None

        try:
            logger.info(f"Fitting scaler on training data (shape: {X_train_flat_scaled.shape})...")
            scaler.fit(X_train_flat_scaled)
            logger.info("Transforming data with fitted scaler...")
            X_train_flat_scaled = scaler.transform(X_train_flat_scaled)
            X_val_flat_scaled = scaler.transform(X_val_flat_scaled)
            X_test_flat_scaled = scaler.transform(X_test_flat_scaled)

            # Check AFTER scaling
            if np.isnan(X_train_flat_scaled).any() or np.isinf(X_train_flat_scaled).any() or \
               np.isnan(X_val_flat_scaled).any() or np.isinf(X_val_flat_scaled).any() or \
               np.isnan(X_test_flat_scaled).any() or np.isinf(X_test_flat_scaled).any():
                logger.error("NaN or Infinity found AFTER scaling! Check data or scaler. Returning unscaled data for safety.")
                # Return original unscaled data to prevent downstream errors with NaNs/Infs
                return X_train_flat_df.values, X_val_flat_df.values, X_test_flat_df.values, None 
            logger.info("Feature normalization complete.")

            if save_path and scaler:
                try:
                    joblib.dump(scaler, save_path)
                    logger.info(f"Saved feature scaler to {save_path}")
                except Exception as e:
                    logger.error(f"Error saving scaler to {save_path}: {e}")
        
        except Exception as e:
            logger.error(f"Error during scaling: {e}. Returning unscaled data.", exc_info=True)
            return X_train_flat_df.values, X_val_flat_df.values, X_test_flat_df.values, None

    else:
        logger.info("Skipping feature normalization as per arguments.")

    return X_train_flat_scaled, X_val_flat_scaled, X_test_flat_scaled, scaler
# +++ End Helper: Scale Features +++

# +++ Helper: Clip DataFrame Features +++
def clip_dataframe_features(df_train, df_val, df_test, lower_q=0.01, upper_q=0.99):
    """Clips features in train, val, and test DataFrames based on train set quantiles."""
    logger.info(f"Clipping features based on training data quantiles: lower_q={lower_q}, upper_q={upper_q}")
    df_train_clipped = df_train.copy()
    df_val_clipped = df_val.copy()
    df_test_clipped = df_test.copy()

    for col in df_train.columns:
        if pd.api.types.is_numeric_dtype(df_train[col]):
            lower_bound = df_train[col].quantile(lower_q)
            upper_bound = df_train[col].quantile(upper_q)
            
            # Clip train
            df_train_clipped[col] = df_train_clipped[col].clip(lower_bound, upper_bound)
            # Clip val (using train bounds)
            if not df_val_clipped.empty and col in df_val_clipped.columns:
                 df_val_clipped[col] = df_val_clipped[col].clip(lower_bound, upper_bound)
            # Clip test (using train bounds)
            if not df_test_clipped.empty and col in df_test_clipped.columns:
                df_test_clipped[col] = df_test_clipped[col].clip(lower_bound, upper_bound)
        else:
            logger.warning(f"Column '{col}' is not numeric. Skipping clipping for this column.")
            
    logger.info("Feature clipping complete.")
    return df_train_clipped, df_val_clipped, df_test_clipped
# +++ End Helper: Clip DataFrame Features +++


# +++ Optuna Objective Function +++
def objective(trial, args, current_run_specific_dir):
    # Convert trial number to a zero-padded string for consistent file naming
    trial_num_str = str(trial.number).zfill(3)
    
    # Create a unique directory for this trial's artifacts (model, plots, metrics)
    # The directory name includes the trial number, symbol, model type, and a timestamp.
    timestamp_for_trial_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_dir_name = f"trial_{trial_num_str}_{args.symbol}_cnntransformer_single_stack_{timestamp_for_trial_dir}"
    trial_path = Path(current_run_specific_dir) / trial_dir_name
    
    # Ensure trial_path (directory for this trial's artifacts) exists
    trial_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Trial {trial.number}: Artifacts will be saved to: {trial_path}")

    # --- Hyperparameter suggestion or loading fixed params ---
    if args.no_optuna:
        logger.info(f"Trial {trial.number} (single run mode / --no-optuna): Using fixed hyperparameters from CLI arguments.")
        # Convert MLP units string to list of ints
        fixed_mlp_units_list = [int(u.strip()) for u in args.mlp_units_fixed.split(',')]

        hp = {
            # CNN specific
            'cnn_filters_1': args.cnn_filters_1_fixed,
            'cnn_kernel_size_1': args.cnn_kernel_size_1_fixed,
            'cnn_pool_size': args.cnn_pool_size_fixed,
            'cnn_pool_strides': args.cnn_pool_strides_fixed,
            'cnn_dropout_rate': args.cnn_dropout_rate_fixed, # Added from fixed args

            # Embedding specific
            'cat_embedding_dim_factor': args.cat_embedding_dim_factor_fixed,

            # Transformer specific
            'num_transformer_blocks': args.num_transformer_blocks_fixed,
            'num_heads': args.num_heads_fixed,
            'head_size': args.head_size_fixed,
            'ff_dim_factor': args.ff_dim_factor_fixed,
            'transformer_dropout_rate': args.dropout_rate_fixed, # Use general dropout for transformer

            # MLP specific
            'mlp_units': fixed_mlp_units_list,
            'mlp_dropout_rate': args.mlp_dropout_rate_fixed,

            # Training & Loss specific
            'learning_rate': args.learning_rate_fixed,
            'weight_decay': args.weight_decay_fixed,
            'batch_size': args.batch_size_fixed,
            'focal_alpha': args.focal_alpha_fixed,
            'focal_gamma': args.focal_gamma_fixed,
            
            # Instance Norm
            'use_instance_norm': args.use_instance_norm, # Get from main args
            'instance_norm_affine': args.instance_norm_affine_fixed, # Get from fixed args

            # +++ Add normalization_type for --no-optuna mode +++
            'normalization_type': args.normalization if not args.use_instance_norm else "none", # Correctly set normalization_type
            'ffn_activation': 'gelu'  # <<< ADDED DEFAULT FFN ACTIVATION HERE
        }
        # Update the trial object with these fixed parameters for Optuna's DB record,
        # even though they are not being suggested. This makes the DB self-contained.
        # Optuna doesn't have a direct way to just "log" params for a no_optuna trial,
        # so we suggest them as fixed values.
        for p_name, p_value in hp.items():
            if isinstance(p_value, (int, float, str)): # Optuna only likes primitive types here
                 trial.set_user_attr(p_name, p_value) # Store in user attrs for clarity
            elif isinstance(p_value, list) and p_name == 'mlp_units': # Handle mlp_units specifically if needed for logging
                 trial.set_user_attr(p_name, json.dumps(p_value)) # Store as JSON string
            # Other types might need special handling or can be skipped for DB logging if complex

    else: # Optuna is active
        logger.info(f"Trial {trial.number} (Optuna mode): Suggesting hyperparameters.")
        hp = {
            # CNN parameters for the single stack
            # 'cnn_filters_1': trial.suggest_int('cnn_filters_1', 32, 128, step=16), # OLD - will be replaced by cnn_base_filters
            # 'cnn_kernel_size_1': trial.suggest_int('cnn_kernel_size_1', 3, 9, step=2), # OLD - kernel fixed to 3
            'cnn_base_filters': trial.suggest_int('cnn_base_filters', 32, 128, step=16), # NEW - for the 3-layer stack
            # +++ ADD SUGGESTIONS FOR POOLING PARAMS +++
            # Strided conv is now default, so pool_size/strides might not be directly tuned unless we offer MaxPool as an option
            'cnn_dropout_rate': trial.suggest_float('cnn_dropout_rate', 0.05, 0.3, step=0.05),


            # Embedding parameters (example: if we were tuning categorical embedding output dimension factor)
            'cat_embedding_dim_factor': trial.suggest_categorical('cat_embedding_dim_factor', [0.25, 0.5, 0.75, 1.0]),

            # Transformer parameters
            'num_transformer_blocks': trial.suggest_int('num_transformer_blocks', 1, 3), # o3: 1-3
            'num_heads': trial.suggest_int('num_heads', 1, 4), # o3: 1-4
            'head_size': trial.suggest_categorical('head_size', [32, 48, 64, 96]), # o3: [32,48,64,96]
            'ff_dim_factor': trial.suggest_categorical('ff_dim_factor', [2.0, 4.0]), # o3: [2.0, 4.0]
            'ffn_activation': trial.suggest_categorical('ffn_activation', ['relu', 'gelu', 'swish']), # Suggested ffn_activation
            
            # MLP head parameters
            # For simplicity, let's use a fixed structure for MLP units in Optuna for now, or suggest number of layers and units per layer
            'mlp_num_layers': trial.suggest_int('mlp_num_layers', 1, 3), # e.g., 1 to 3 layers
            'mlp_units_per_layer': trial.suggest_categorical('mlp_units_per_layer', [32, 64, 128]), # units for each layer
            # mlp_units will be constructed based on mlp_num_layers and mlp_units_per_layer

            # Dropout rates
            'transformer_dropout_rate': trial.suggest_float('transformer_dropout_rate', 0.10, 0.30, step=0.05), # o3: 0.15-0.40 (adjusted range slightly)
            'mlp_dropout_rate': trial.suggest_float('mlp_dropout_rate', 0.10, 0.40, step=0.05), # o3: tied to general dropout

            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 5e-6, 5e-5, log=True), # o3: 5e-6 to 5e-5
            'weight_decay': trial.suggest_float('weight_decay', 5e-5, 5e-4, log=True), # o3: 5e-5 to 5e-4
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]), # o3: [32, 64, 128]

            # Focal Loss parameters
            'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.5, step=0.05), # o3: 0.1-0.5
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.5), # o3: 1.0-3.0
            
            # Instance Norm (fixed as per research, but could be tuned if desired)
            'use_instance_norm': trial.suggest_categorical('use_instance_norm', [True]), # Fixed to True
            'instance_norm_affine': trial.suggest_categorical('instance_norm_affine', [True]), # Fixed to True
            
            # Add normalization_type, derived from args, as it's not an Optuna-tuned HP here
            'normalization_type': args.normalization if not trial.params.get('use_instance_norm', args.use_instance_norm) else "none"
        }
        # Construct mlp_units based on Optuna suggestions
        hp['mlp_units'] = [hp['mlp_units_per_layer']] * hp['mlp_num_layers']


    # +++ Define cfg dictionary using hp and args BEFORE dataset creation +++
    # This dictionary will store ALL parameters for this trial run for reproducibility.
    # Start with a copy of all command-line arguments
    cfg = vars(args).copy()
    
    # Add hyperparameter values (either suggested by Optuna or fixed from CLI)
    cfg.update(hp) # hp contains the hyperparameters for the current trial

    # Add trial-specific context
    cfg['trial_number_for_paths'] = trial.number # Store trial number for path construction if needed later

    # Add any other relevant info that might not be in args or hp directly but is trial-specific
    # For example, if d_model is derived:
    d_model_transformer_mha = (hp.get('num_heads', args.num_heads_fixed if args.no_optuna else args.num_heads) *
                               hp.get('head_size', args.head_size_fixed if args.no_optuna else args.head_size))
    cfg['d_model_transformer_mha'] = d_model_transformer_mha # Log the calculated d_model
    
    # Calculate ff_dim based on whether Optuna is active or not
    ff_dim_factor_to_use = hp.get('ff_dim_factor', args.ff_dim_factor_fixed if args.no_optuna else args.ff_dim_factor)
    cfg['transformer_ff_dim'] = int(d_model_transformer_mha * ff_dim_factor_to_use) # Log calculated ff_dim

    # Ensure critical CNN parameters are derived correctly for the new stack
    if args.no_optuna:
        # For fixed runs, cnn_layer_configs is built based on cnn-filters-1-fixed and cnn-kernel-size-1-fixed
        # Assuming a 3-layer stack with increasing dilation as per recent changes
        cfg['cnn_layer_configs'] = [
            {"filters": args.cnn_filters_1_fixed, "kernel": args.cnn_kernel_size_1_fixed, "dilation": 1},
            {"filters": args.cnn_filters_1_fixed, "kernel": args.cnn_kernel_size_1_fixed, "dilation": 2},
            {"filters": args.cnn_filters_1_fixed, "kernel": args.cnn_kernel_size_1_fixed, "dilation": 4},
        ]
        # These are directly used by build_model, no need for separate cnn_base_filters_fixed if structure is defined here
        cfg['cnn_pool_size'] = args.cnn_pool_size_fixed
        cfg['cnn_pool_strides'] = args.cnn_pool_strides_fixed
        cfg['cnn_activation'] = args.cnn_activation # From args, fixed for Optuna anyway
        cfg['cnn_use_bias'] = not args.cnn_no_bias # from args
    else: # Optuna mode
        # cnn_base_filters is suggested by Optuna
        # kernel size is fixed at 3, dilations (1,2,4) are fixed for the 3-layer stack
        cfg['cnn_layer_configs'] = [
            {"filters": hp['cnn_base_filters'], "kernel": 3, "dilation": 1},
            {"filters": hp['cnn_base_filters'], "kernel": 3, "dilation": 2},
            {"filters": hp['cnn_base_filters'], "kernel": 3, "dilation": 4},
        ]
        # Pooling and activation are fixed as per research for Optuna runs
        cfg['cnn_pool_size'] = 2 # Strided conv kernel=2 effectively
        cfg['cnn_pool_strides'] = 2 # Strided conv strides=2
        cfg['cnn_activation'] = 'gelu' # Fixed for Optuna
        cfg['cnn_use_bias'] = not args.cnn_no_bias # from args (can be fixed true/false if preferred)
        # cnn_dropout_rate is already in hp

    # +++ ADD CURRENT UTC TIMESTAMP +++
    cfg['run_timestamp_utc'] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Trial {trial.number}: Hyperparameters (cfg) for this trial (defined before data loading): {json.dumps(cfg, indent=2, cls=NpEncoder)}") # Using NpEncoder for safety

    # --- Data Loading and Preprocessing ---
    logger.info(f"Trial {trial.number}: Loading features from {args.feature_parquet}...")
    try:
        df = pd.read_parquet(args.feature_parquet)
        logger.info(f"Trial {trial.number}: Loaded data. Shape: {df.shape}")

        if args.smoke_test_nrows is not None:
            if args.smoke_test_nrows > 0 and args.smoke_test_nrows < len(df):
                df = df.head(args.smoke_test_nrows)
                logger.info(f"Trial {trial.number}: SMOKE TEST active. Truncated DataFrame to first {args.smoke_test_nrows} rows. New shape: {df.shape}")
            elif args.smoke_test_nrows >= len(df):
                logger.warning(f"Trial {trial.number}: SMOKE TEST --smoke-test-nrows ({args.smoke_test_nrows}) is >= df length ({len(df)}). Using full loaded data for smoke test.")
            else: # <= 0
                logger.warning(f"Trial {trial.number}: SMOKE TEST --smoke-test-nrows ({args.smoke_test_nrows}) is invalid (<=0). Ignoring and using full loaded data for smoke test.")


        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Trial {trial.number}: DataFrame index is not a DatetimeIndex. Type: {type(df.index)}. Attempting to set 'open_timestamp' or 'timestamp'.")
            if 'open_timestamp' in df.columns:
                df['open_timestamp'] = pd.to_datetime(df['open_timestamp'])
                df = df.set_index('open_timestamp')
                logger.info(f"Trial {trial.number}: Set 'open_timestamp' as DatetimeIndex.")
            elif 'timestamp' in df.columns: # Fallback
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                logger.info(f"Trial {trial.number}: Set 'timestamp' as DatetimeIndex.")
            else:
                logger.error(f"Trial {trial.number}: Could not find a suitable timestamp column to set as DatetimeIndex.")
                raise ValueError("DataFrame must have a DatetimeIndex or a suitable timestamp column.")
        
        df = df.sort_index()
        logger.info(f"Trial {trial.number}: DataFrame sorted by index.")

    except FileNotFoundError:
        logger.error(f"Trial {trial.number}: Feature file not found at {args.feature_parquet}. Exiting trial.")
        # Optuna interprets a returned value like 0.0 or raising TrialPruned for failed trials.
        # Depending on Optuna version and setup, returning a specific value might be preferred for failure vs. pruning.
        # For now, returning 0.0 as a poor score.
        return 0.0 
    except Exception as e:
        logger.error(f"Trial {trial.number}: Error loading data: {e}", exc_info=True)
        return 0.0

    # --- 1a. Handle Target NaNs ---
    if TARGET_COLUMN_NAME not in df.columns:
        logger.error(f"Trial {trial.number}: Crucial target column '{TARGET_COLUMN_NAME}' not found in DataFrame after loading. Columns: {df.columns.tolist()}")
        return 0.0 # Cannot proceed without target

    initial_rows = len(df)
    df.dropna(subset=[TARGET_COLUMN_NAME], inplace=True)
    rows_after_dropna = len(df)
    logger.info(f"Trial {trial.number}: Dropped {initial_rows - rows_after_dropna} rows with NaN in target column '{TARGET_COLUMN_NAME}'. Shape after drop: {df.shape}")
    
    if df.empty:
        logger.error(f"Trial {trial.number}: DataFrame is empty after dropping NaN targets. Cannot proceed.")
        return 0.0

    # --- 2. Define Target Variable ---
    # Target is pre-calculated and is TARGET_COLUMN_NAME
    logger.info(f"Trial {trial.number}: Using pre-calculated target column: {TARGET_COLUMN_NAME}")
    target_series = df[TARGET_COLUMN_NAME].astype(float) # Ensure target is float for ML

    # --- 3. Define Feature Sets based on args and available columns ---
    all_available_columns = df.columns.tolist()
    selected_feature_columns = [] # Initialize

    if args.feature_list_file:
        logger.info(f"Trial {trial.number}: Loading features from file: {args.feature_list_file}")
        try:
            with open(args.feature_list_file, 'r') as f:
                features_from_file = [line.strip() for line in f if line.strip()]
            
            validated_features = []
            missing_features = []
            for feature_name in features_from_file:
                if feature_name in all_available_columns:
                    validated_features.append(feature_name)
                else:
                    missing_features.append(feature_name)
            
            if missing_features:
                logger.warning(f"Trial {trial.number}: The following features from {args.feature_list_file} were NOT found in the DataFrame and will be SKIPPED: {missing_features}")
            
            if not validated_features:
                logger.error(f"Trial {trial.number}: No valid features found from {args.feature_list_file} in the DataFrame. Cannot proceed.")
                return 0.0 # Or raise optuna.exceptions.TrialPruned
            
            selected_feature_columns = validated_features
            features_df = df[selected_feature_columns].copy() # +++ Assign to features_df directly +++
            logger.info(f"Trial {trial.number}: Successfully loaded and validated {len(selected_feature_columns)} features from {args.feature_list_file}. These will be used exclusively.")
            logger.info(f"Trial {trial.number}: --use-short-term-features, --use-medium-term-features, --use-long-term-features, --use-regime-features flags will be IGNORED due to --feature-list-file.")

        except FileNotFoundError:
            logger.error(f"Trial {trial.number}: Feature list file not found at {args.feature_list_file}. Exiting trial.")
            return 0.0 
        except Exception as e_feat_file:
            logger.error(f"Trial {trial.number}: Error processing feature list file {args.feature_list_file}: {e_feat_file}. Exiting trial.")
            return 0.0
    else:
        logger.info(f"Trial {trial.number}: No --feature-list-file provided. Using feature selection flags (--use-short-term-features, etc.).")
        # --- Original feature selection logic based on --use-xxx-term-features flags ---
        short_term_features_auto = []
        medium_term_features_auto = []
        long_term_features_auto = []
        regime_features_auto = []

        known_non_feature_cols = [
            TARGET_COLUMN_NAME, 'timestamp', 'open_timestamp', 'close_timestamp', # Index or target or other time related
            'open_time_tb_1min', 'close_time_tb_1min', 'timestamp_x_tb_1min', # From specific datasets
            'open', 'high', 'low', 'close', 'volume', # Raw OHLCV typically not direct features unless transformed
            # Add any other known target variations or non-feature columns if they exist
            'target', 'sl', 'tp', 'target_mid_price_pct_change_4h', 'target_fixed_horizon_1h',
            'target_adaptive_atr_1.5_14_1.0', 'target_fixed_1.0_0.5', 'target_fixed_2.0_1.0' 
        ]
        # Extend with common suffixes for other potential targets if any standard naming is used.
        known_non_feature_cols.extend([col for col in all_available_columns if col.startswith('target_')])

        temp_df_target = df[[TARGET_COLUMN_NAME]] # For add_features_to_selection

        # Helper function to add features to the selection lists
        def add_features_to_selection(feature_list_auto, df_source, current_combined_list, temp_df_target, category_name):
            # Original logic from the script, ensure it's consistent
            # This is a placeholder for the actual add_features_to_selection logic from the script
            # It would typically iterate through df_source.columns, check patterns, and append to feature_list_auto
            # For example:
            for col in df_source.columns:
                if col not in current_combined_list and col not in temp_df_target.columns and col not in known_non_feature_cols:
                    if category_name == "short" and ("_tb_" in col or "_st_" in col or "_m1_" in col or "_m5_" in col or "_m15_" in col): # Example patterns
                        feature_list_auto.append(col)
                    elif category_name == "medium" and ("_mt_" in col or "_h1_" in col or "_h4_" in col): # Example patterns
                         feature_list_auto.append(col)
                    elif category_name == "long" and ("_lt_" in col or "_d1_" in col): # Example patterns
                         feature_list_auto.append(col)
                    elif category_name == "regime" and ("_rgm_" in col or "market_regime" in col): # Example patterns
                         feature_list_auto.append(col)
            return sorted(list(set(feature_list_auto)))


        # Populate feature lists (simplified representation of the original logic)
        # The original script had more complex logic for this, including add_features_to_selection
        # This part needs to accurately reflect how the original script decided what goes into these _auto lists.
        # For now, this is a conceptual placeholder.
        
        # Rough conceptual population based on common suffixes - replace with actual script logic if different
        for col in all_available_columns:
            if col in known_non_feature_cols or col == TARGET_COLUMN_NAME:
                continue
            if "_tb_" in col or "_st_" in col or any(suffix in col for suffix in ["_m1_", "_m5_", "_m15_", "_m30_"]):
                short_term_features_auto.append(col)
            elif "_mt_" in col or any(suffix in col for suffix in ["_h1_", "_h4_", "_h6_", "_h12_"]):
                medium_term_features_auto.append(col)
            elif "_lt_" in col or any(suffix in col for suffix in ["_d1_", "_wk_", "_mn_"]):
                long_term_features_auto.append(col)
            elif "_rgm_" in col or "market_regime" in col or "vol_regime" in col:
                regime_features_auto.append(col)

        short_term_features_auto = sorted(list(set(short_term_features_auto)))
        medium_term_features_auto = sorted(list(set(medium_term_features_auto)))
        long_term_features_auto = sorted(list(set(long_term_features_auto)))
        regime_features_auto = sorted(list(set(regime_features_auto)))
        
        logger.info(f"Trial {trial.number}: Auto-detected feature counts - Short: {len(short_term_features_auto)}, Medium: {len(medium_term_features_auto)}, Long: {len(long_term_features_auto)}, Regime: {len(regime_features_auto)}")

        combined_selected_features = []
        if args.use_short_term_features:
            combined_selected_features.extend(short_term_features_auto)
            logger.info(f"Trial {trial.number}: Including {len(short_term_features_auto)} short-term features.")
        if args.use_medium_term_features:
            combined_selected_features.extend(medium_term_features_auto)
            logger.info(f"Trial {trial.number}: Including {len(medium_term_features_auto)} medium-term features.")
        if args.use_long_term_features:
            combined_selected_features.extend(long_term_features_auto)
            logger.info(f"Trial {trial.number}: Including {len(long_term_features_auto)} long-term features.")
        if args.use_regime_features:
            combined_selected_features.extend(regime_features_auto)
            logger.info(f"Trial {trial.number}: Including {len(regime_features_auto)} regime features.")
        
        selected_feature_columns = sorted(list(set(combined_selected_features)))


    # --- After selected_feature_columns is populated either by file or by flags ---
    if not selected_feature_columns:
        logger.error(f"Trial {trial.number}: No features were selected based on CLI arguments and available DataFrame columns. Exiting trial.")
        logger.info(f"Review CLI flags: --use-short-term-features={args.use_short_term_features}, etc., and ensure consolidated Parquet file columns are named with expected suffixes.")
        return 0.0 # Return a poor score for Optuna
        
    # Remove duplicate columns that might have arisen from concat, preserving the first occurrence
    # features_df = selected_features_intermediate_df.loc[:,~selected_features_intermediate_df.columns.duplicated(keep='first')] # --- REMOVE THIS LINE ---
    # If --feature-list-file was used, features_df is already defined with selected columns.
    # If flags were used, features_df needs to be df[selected_feature_columns]
    if not args.feature_list_file: # If flags were used, now subset the main df
        features_df = df[selected_feature_columns].copy()
    
    # Deduplication should happen on features_df if necessary, though selected_feature_columns should be unique already
    features_df = features_df.loc[:,~features_df.columns.duplicated(keep='first')]

    # Update combined_feature_list to match the columns and order in features_df (after deduplication)
    combined_feature_list = features_df.columns.tolist()

    logger.info(f"Trial {trial.number}: Final selected features for model input: {len(combined_feature_list)}. Shape of features_df: {features_df.shape}")
    if not combined_feature_list:
        logger.error(f"Trial {trial.number}: 'combined_feature_list' is empty after all selection and deduplication steps. Cannot proceed.")
        return 0.0

    # --- 3a. Define Categorical Features and Features to Exclude from Scaling ---
    # These are based on the final `combined_feature_list`
    features_to_exclude_from_scaling = []
    categorical_feature_specs = {} # Format: {'feat_name': {'input_dim': N, 'output_dim': M}}

    logger.info(f"Trial {trial.number}: Identifying features for scaling exclusion and categorical embedding from {len(combined_feature_list)} selected features.")
    for col_name in combined_feature_list: 
        # Candlestick patterns (conventionally start with CDL)
        if col_name.upper().startswith('CDL'):
            if col_name not in features_to_exclude_from_scaling:
                 features_to_exclude_from_scaling.append(col_name)
        
        # Binary stochastic flags (e.g., is_stoch_ob_80_tb_15min_tb_15min)
        elif col_name.startswith('is_stoch_'): # ADDED THIS ELIF
            if col_name not in features_to_exclude_from_scaling:
                features_to_exclude_from_scaling.append(col_name)

        # Divergence features (check for 'DIVERGENCE' and type like 'BULLISH', 'BEARISH', 'HIDDEN')
        # These are often binary or ternary flags and should not be scaled.
        is_divergence_feature = 'DIVERGENCE' in col_name.upper() and \
                                any(term in col_name.upper() for term in ['BULLISH', 'BEARISH', 'HIDDEN'])
        if is_divergence_feature:
            if col_name not in features_to_exclude_from_scaling:
                features_to_exclude_from_scaling.append(col_name)
            # Optionally, if these divergence flags could have more than 2 states and are not purely binary,
            # they could be considered for embedding if their cardinality is small.
            # Example: 
            # unique_div_vals = features_df[col_name].nunique()
            # if unique_div_vals > 2 and unique_div_vals <= 5: # e.g., -1, 0, 1 
            #    cat_input_dim = unique_div_vals
            #    cat_output_dim = max(1, min(unique_div_vals // 2, 5)) # Small embedding
            #    categorical_feature_specs[col_name] = {'input_dim': cat_input_dim, 'output_dim': cat_output_dim}
            #    logger.info(f"Treating divergence feature '{col_name}' as categorical for embedding: in={cat_input_dim}, out={cat_output_dim}")

        # Market Regime features (e.g., market_regime_tb_1min) are categorical and should be embedded.
        is_regime_col_final = bool(re.match(r"market_regime_.*", col_name, re.IGNORECASE))
        if is_regime_col_final:
            if col_name not in features_to_exclude_from_scaling:
                features_to_exclude_from_scaling.append(col_name) # Regime features are typically not scaled
            
            if col_name not in categorical_feature_specs: # Avoid re-processing if already handled (e.g. if a divergence was also regime)
                unique_values_in_col = features_df[col_name].dropna().unique()
                num_unique_categories = len(unique_values_in_col)
                
                if num_unique_categories == 0:
                    logger.warning(f"Trial {trial.number}: Categorical feature candidate '{col_name}' has no unique values after dropna(). Skipping for embedding.")
                    if col_name in features_to_exclude_from_scaling: features_to_exclude_from_scaling.remove(col_name)
                    continue
                if num_unique_categories == 1:
                     logger.warning(f"Trial {trial.number}: Categorical feature '{col_name}' has only 1 unique value: {unique_values_in_col}. This is not useful for embedding. It will be passed as is or scaled if not excluded.")
                     # If it has only one value, it's constant after NaNs are handled. Scaling won't harm but embedding is pointless.
                     # Keep it in exclude_from_scaling if it was already there (e.g. regime)
                
                # Determine embedding input and output dimensions
                cat_input_dim = num_unique_categories
                # Use a hyperparameter for the embedding output dimension factor
                cat_output_dim = max(1, int(cat_input_dim * hp['cat_embedding_dim_factor'])) 
                cat_output_dim = min(cat_output_dim, 50) # Cap embedding output dimension to a reasonable max (e.g., 50)
                cat_output_dim = max(1, cat_output_dim) # Ensure at least 1D embedding

                categorical_feature_specs[col_name] = {
                    'input_dim': cat_input_dim, 
                    'output_dim': cat_output_dim
                }
                logger.info(f"Trial {trial.number}: Defined categorical feature '{col_name}': input_dim={cat_input_dim}, output_dim={cat_output_dim}. Unique values (up to 5): {unique_values_in_col[:5]}")

    logger.info(f"Trial {trial.number}: Total features to EXCLUDE from scaling ({len(features_to_exclude_from_scaling)}): {features_to_exclude_from_scaling}")
    logger.info(f"Trial {trial.number}: Total CATEGORICAL features for embedding ({len(categorical_feature_specs)}): {list(categorical_feature_specs.keys())}")
    # For debugging, log the specs themselves
    # for cat_feat_name, cat_spec_detail in categorical_feature_specs.items():
    #    logger.debug(f"Categorical Spec - {cat_feat_name}: {cat_spec_detail}")

    # --- 4. Impute Missing Values (before splitting, scaling, and sequence creation) ---
    if features_df.isnull().any().any():
        logger.info(f"Trial {trial.number}: NaNs found in features_df (Shape: {features_df.shape}). Starting NaN imputation...")
        features_df_imputed = features_df.copy()

        # Step 1: Forward-fill Volume Profile columns
        vp_columns = [col for col in features_df_imputed.columns if col.startswith('vp_')]
        if vp_columns:
            logger.info(f"Trial {trial.number}: Forward-filling NaNs for {len(vp_columns)} Volume Profile columns: {vp_columns[:5]}...")
            initial_nans_in_vp = features_df_imputed[vp_columns].isnull().sum().sum()
            features_df_imputed[vp_columns] = features_df_imputed[vp_columns].ffill()
            nans_after_ffill_in_vp = features_df_imputed[vp_columns].isnull().sum().sum()
            logger.info(f"Trial {trial.number}: VP columns ffill complete. NaNs reduced from {initial_nans_in_vp} to {nans_after_ffill_in_vp} in VP columns.")
            if nans_after_ffill_in_vp > 0:
                logger.info(f"Trial {trial.number}: {nans_after_ffill_in_vp} NaNs remain in VP columns (likely at the beginning), will be handled by median imputation if applicable.")

        # Step 2: Median imputation for any remaining NaNs in any column
        logger.info(f"Trial {trial.number}: Applying median imputation for any remaining NaNs across all columns...")
        remaining_nan_cols_before_median = features_df_imputed.columns[features_df_imputed.isnull().any()].tolist()
        if remaining_nan_cols_before_median:
            logger.info(f"Trial {trial.number}: Columns with NaNs before median imputation: {remaining_nan_cols_before_median[:10]}...")
            for col in features_df_imputed.columns: # Iterate over all columns for median imputation
                if features_df_imputed[col].isnull().any():
                    median_val = features_df_imputed[col].median()
                    if pd.isna(median_val):
                        logger.warning(f"Trial {trial.number}: Median for column '{col}' is NaN (e.g., all NaNs). Imputing with 0.")
                        features_df_imputed[col] = features_df_imputed[col].fillna(0)
                    else:
                        features_df_imputed[col] = features_df_imputed[col].fillna(median_val)
            logger.info(f"Trial {trial.number}: Median imputation step finished.")
        else:
            logger.info(f"Trial {trial.number}: No NaNs remaining after VP ffill (if any). Skipping median imputation loop.")

        # Verify final imputation
        if features_df_imputed.isnull().any().any():
            final_nan_cols = features_df_imputed.columns[features_df_imputed.isnull().any()].tolist()
            logger.error(f"Trial {trial.number}: NaNs still present after all imputation steps. Columns with NaNs: {final_nan_cols}. Pruning.")
            raise optuna.exceptions.TrialPruned("NaNs present after all imputation.")
        else:
            logger.info(f"Trial {trial.number}: All NaN imputation complete. No NaNs remain.")
        
        features_df = features_df_imputed
    else:
        logger.info(f"Trial {trial.number}: No NaNs found in features_df initially. Skipping imputation.")


    # --- 4a. Data Splitting (Chronological) ---
    # Ensure features_df and target_series are aligned by index before splitting.
    # They should be aligned if features_df was derived from df and target_series from df.
    # A quick check:
    if not features_df.index.equals(target_series.index):
        logger.error(f"Trial {trial.number}: Index mismatch between features_df and target_series before splitting. This is critical. Pruning.")
        # Log index details for debugging
        logger.error(f"features_df index (first 5): {features_df.index[:5]}, target_series index (first 5): {target_series.index[:5]}")
        logger.error(f"features_df index (last 5): {features_df.index[-5:]}, target_series index (last 5): {target_series.index[-5:]}")
        raise optuna.exceptions.TrialPruned("Index mismatch between features and target before split.")

    n_samples = len(features_df)
    if n_samples == 0:
        logger.error(f"Trial {trial.number}: features_df is empty before splitting. Cannot proceed.")
        return 0.0 # Or raise optuna.exceptions.TrialPruned

    test_split_idx = int(n_samples * (1 - args.test_size))
    
    # Split features_df (which is now imputed)
    X_train_val_flat_df = features_df.iloc[:test_split_idx]
    X_test_flat_df = features_df.iloc[test_split_idx:]
    
    # Split target_series
    y_train_val_flat = target_series.iloc[:test_split_idx]
    y_test_flat = target_series.iloc[test_split_idx:]

    # Further split train_val into train and validation
    n_train_val_samples = len(X_train_val_flat_df)
    val_split_idx = int(n_train_val_samples * (1 - args.val_size)) # val_size is proportion of train_val for validation

    X_train_flat_df = X_train_val_flat_df.iloc[:val_split_idx]
    X_val_flat_df = X_train_val_flat_df.iloc[val_split_idx:]
    
    y_train_flat = y_train_val_flat.iloc[:val_split_idx]
    y_val_flat = y_train_val_flat.iloc[val_split_idx:]

    logger.info(f"Trial {trial.number}: Data split. Train: {X_train_flat_df.shape}, Val: {X_val_flat_df.shape}, Test: {X_test_flat_df.shape}")
    logger.info(f"Trial {trial.number}: Target shapes. Train: {y_train_flat.shape}, Val: {y_val_flat.shape}, Test: {y_test_flat.shape}")

    if X_train_flat_df.empty or X_val_flat_df.empty:
        logger.error(f"Trial {trial.number}: Training or Validation DataFrame is empty after split. Pruning.")
        logger.error(f"Train shape: {X_train_flat_df.shape}, Val shape: {X_val_flat_df.shape}, Test shape: {X_test_flat_df.shape}")
        logger.error(f"Original n_samples: {n_samples}, test_split_idx: {test_split_idx}, n_train_val_samples: {n_train_val_samples}, val_split_idx: {val_split_idx}")
        raise optuna.exceptions.TrialPruned("Empty train/val set after split.")
    
    # Store the final ordered list of feature names AFTER all processing (imputation, selection)
    # This list is crucial for the model's input layer to correctly map raw input to embeddings/numerical paths.
    # `combined_feature_list` should already reflect the columns in `features_df`.
    all_selected_features_ordered = combined_feature_list 
    logger.info(f"Trial {trial.number}: Final ordered list of {len(all_selected_features_ordered)} features for model input (post-imputation, pre-scaling): {all_selected_features_ordered[:5]}...")


    # --- 5. Feature Scaling and Clipping ---
    scaler_save_path = trial_path / f"feature_scaler_trial_{trial.number}.joblib" if trial_path and args.save_results else None
    
    # features_to_exclude_from_scaling was defined earlier based on combined_feature_list
    # Ensure these lists correctly reference columns in X_train_flat_df
    cols_to_scale = [col for col in X_train_flat_df.columns if col not in features_to_exclude_from_scaling and col in all_selected_features_ordered]
    cols_excluded_from_scaling_present_in_df = [col for col in X_train_flat_df.columns if col in features_to_exclude_from_scaling and col in all_selected_features_ordered]

    logger.info(f"Trial {trial.number}: Identified {len(cols_to_scale)} features for scaling: {cols_to_scale[:5]}...")
    logger.info(f"Trial {trial.number}: Identified {len(cols_excluded_from_scaling_present_in_df)} features to EXCLUDE from scaling (and are present): {cols_excluded_from_scaling_present_in_df[:5]}...")
    
    # Initialize NumPy arrays to hold potentially scaled data.
    # These will be populated correctly whether scaling happens or not.
    # We create copies to avoid modifying the original DataFrames if scaling is skipped or fails.
    X_train_processed_np = X_train_flat_df.copy().to_numpy()
    X_val_processed_np = X_val_flat_df.copy().to_numpy()
    X_test_processed_np = X_test_flat_df.copy().to_numpy()
    
    # Clipping: Applied to a *copy* of the DFs to avoid modifying originals if scaling also happens.
    # And applied only to the columns that will be scaled.
    if args.clip_features and cols_to_scale:
        logger.info(f"Trial {trial.number}: Applying clipping to {len(cols_to_scale)} features before scaling.")
        # Create copies for clipping to avoid modifying the original DFs that might be used by scaling later
        X_train_flat_df_for_clip_scale = X_train_flat_df.copy()
        X_val_flat_df_for_clip_scale = X_val_flat_df.copy()
        X_test_flat_df_for_clip_scale = X_test_flat_df.copy()

        # Perform clipping on the copies for the specific columns
        # The clip_dataframe_features function returns new DataFrames.
        # We need to handle the case where a split might be empty (e.g. X_test_flat_df_for_clip_scale)
        
        # Prepare inputs for clipping. If a df is empty, pass an empty df with same columns.
        train_to_clip = X_train_flat_df_for_clip_scale[cols_to_scale]
        
        val_to_clip = pd.DataFrame(columns=cols_to_scale) # Default to empty
        if not X_val_flat_df_for_clip_scale.empty and all(c in X_val_flat_df_for_clip_scale.columns for c in cols_to_scale):
            val_to_clip = X_val_flat_df_for_clip_scale[cols_to_scale]
            
        test_to_clip = pd.DataFrame(columns=cols_to_scale) # Default to empty
        if not X_test_flat_df_for_clip_scale.empty and all(c in X_test_flat_df_for_clip_scale.columns for c in cols_to_scale):
            test_to_clip = X_test_flat_df_for_clip_scale[cols_to_scale]

        clipped_train_cols, clipped_val_cols, clipped_test_cols = clip_dataframe_features(
            train_to_clip, val_to_clip, test_to_clip,
                lower_q=args.clip_lower_q, upper_q=args.clip_upper_q
            )

        # Update the original copied DFs (X_train_flat_df_for_clip_scale, etc.) with the clipped columns
        X_train_flat_df_for_clip_scale[cols_to_scale] = clipped_train_cols
        if not X_val_flat_df_for_clip_scale.empty: X_val_flat_df_for_clip_scale[cols_to_scale] = clipped_val_cols
        if not X_test_flat_df_for_clip_scale.empty: X_test_flat_df_for_clip_scale[cols_to_scale] = clipped_test_cols
        
        # These DFs (now potentially clipped) will be used for scaling.
        df_to_scale_train = X_train_flat_df_for_clip_scale
        df_to_scale_val = X_val_flat_df_for_clip_scale
        df_to_scale_test = X_test_flat_df_for_clip_scale
    else: # No clipping, use the original split DFs for scaling
        df_to_scale_train = X_train_flat_df
        df_to_scale_val = X_val_flat_df
        df_to_scale_test = X_test_flat_df

    # Scaling: Applied to `df_to_scale_train` etc. (which are either original or clipped)
    # Use hp['normalization_type'] from Optuna trial suggestion
    current_normalization_method = cfg['normalization'] 
    logger.info(f"Trial {trial.number}: Optuna selected normalization: {current_normalization_method}")

    if current_normalization_method != 'none' and cols_to_scale:
        scaler = None
        if current_normalization_method == 'standard': scaler = StandardScaler()
        elif current_normalization_method == 'minmax': scaler = MinMaxScaler()
        elif current_normalization_method == 'robust': scaler = RobustScaler()
        elif current_normalization_method == 'power_yeo_johnson': 
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        elif current_normalization_method == 'quantile_normal':
            scaler = QuantileTransformer(output_distribution='normal', random_state=trial.number)
        elif current_normalization_method == 'quantile_uniform':
            scaler = QuantileTransformer(output_distribution='uniform', random_state=trial.number)
        
        if scaler:
            logger.info(f"Trial {trial.number}: Fitting scaler ({current_normalization_method}) on {len(cols_to_scale)} features from training data.")
            
            # --- Ensure cols_to_scale are actually numeric in df_to_scale_train ---
            original_cols_to_scale_count = len(cols_to_scale)
            cols_to_scale = df_to_scale_train[cols_to_scale].select_dtypes(include=np.number).columns.tolist()
            if len(cols_to_scale) < original_cols_to_scale_count:
                logger.warning(f"Trial {trial.number}: Refined cols_to_scale to {len(cols_to_scale)} numeric columns before fitting scaler. Original count was {original_cols_to_scale_count}.")
            # --- End ensure numeric ---

            if not cols_to_scale: # If no numeric columns remain after filtering
                logger.warning(f"Trial {trial.number}: No numeric columns left in cols_to_scale for method {current_normalization_method}. Skipping scaling step.")
                # Fall through to the 'else' block that handles skipped scaling
                X_train_processed_np = df_to_scale_train[all_selected_features_ordered].to_numpy()
                X_val_processed_np = df_to_scale_val[all_selected_features_ordered].to_numpy() if not df_to_scale_val.empty else np.array([]).reshape(0, len(all_selected_features_ordered))
                X_test_processed_np = df_to_scale_test[all_selected_features_ordered].to_numpy() if not df_to_scale_test.empty else np.array([]).reshape(0, len(all_selected_features_ordered))
            else:
                try:
                    scaler.fit(df_to_scale_train[cols_to_scale])
                except ValueError as ve:
                    logger.error(f"Trial {trial.number}: ValueError during scaler.fit() for method {current_normalization_method} (e.g., non-positive values for Box-Cox if it were used, or all-constant data for some transformers). Error: {ve}. Pruning trial.")
                    # Returning a poor score or raising TrialPruned
                    # For simplicity here, we'll log and then the code will likely fail later or return poor results naturally.
                    # Better to prune:
                    raise optuna.exceptions.TrialPruned(f"Scaler fit failed for {current_normalization_method}: {ve}")

                
                # Transform the 'cols_to_scale' in each respective DataFrame (train, val, test)
                # We need to store these transformed NumPy arrays and then reconstruct the full feature array.
                
                # Create temporary NumPy arrays for scaled columns
                scaled_train_cols_np = scaler.transform(df_to_scale_train[cols_to_scale])
                scaled_val_cols_np = scaler.transform(df_to_scale_val[cols_to_scale]) if not df_to_scale_val.empty else np.array([]).reshape(0, len(cols_to_scale))
                scaled_test_cols_np = scaler.transform(df_to_scale_test[cols_to_scale]) if not df_to_scale_test.empty else np.array([]).reshape(0, len(cols_to_scale))

                logger.info(f"Trial {trial.number}: Scaled {len(cols_to_scale)} features using {current_normalization_method}.")
                if scaler_save_path: joblib.dump(scaler, scaler_save_path)

                # Now, reconstruct the full X_train_processed_np, X_val_processed_np, X_test_processed_np
                # by placing the scaled columns back into their original positions,
                # alongside the unscaled columns.
                # The `all_selected_features_ordered` list maintains the correct global order.

                # For X_train_processed_np
                temp_train_df_reconstruct = X_train_flat_df.copy() # Start with original order
                temp_train_df_reconstruct[cols_to_scale] = scaled_train_cols_np
                # Ensure only numeric columns before final conversion
                df_slice_train = temp_train_df_reconstruct[all_selected_features_ordered]
                numeric_cols_train = df_slice_train.select_dtypes(include=np.number).columns
                if len(numeric_cols_train) < len(df_slice_train.columns):
                    non_numeric_dropped = list(set(df_slice_train.columns) - set(numeric_cols_train))
                    logger.warning(f"Trial {trial.number}: Dropping non-numeric columns before final .to_numpy() for train: {non_numeric_dropped}")
                X_train_processed_np = df_slice_train[numeric_cols_train].astype(np.float32).to_numpy()
                
                # For X_val_processed_np
                if not X_val_flat_df.empty:
                    temp_val_df_reconstruct = X_val_flat_df.copy()
                    temp_val_df_reconstruct[cols_to_scale] = scaled_val_cols_np
                    df_slice_val = temp_val_df_reconstruct[all_selected_features_ordered]
                    numeric_cols_val = df_slice_val.select_dtypes(include=np.number).columns
                    if len(numeric_cols_val) < len(df_slice_val.columns):
                        non_numeric_dropped_val = list(set(df_slice_val.columns) - set(numeric_cols_val))
                        logger.warning(f"Trial {trial.number}: Dropping non-numeric columns before final .to_numpy() for val: {non_numeric_dropped_val}")
                    X_val_processed_np = df_slice_val[numeric_cols_val].astype(np.float32).to_numpy()
                else: 
                    X_val_processed_np = np.array([]).reshape(0, len(numeric_cols_train) if numeric_cols_train.any() else 0).astype(np.float32)

                # For X_test_processed_np
                if not X_test_flat_df.empty:
                    temp_test_df_reconstruct = X_test_flat_df.copy()
                    temp_test_df_reconstruct[cols_to_scale] = scaled_test_cols_np
                    df_slice_test = temp_test_df_reconstruct[all_selected_features_ordered]
                    numeric_cols_test = df_slice_test.select_dtypes(include=np.number).columns
                    if len(numeric_cols_test) < len(df_slice_test.columns):
                        non_numeric_dropped_test = list(set(df_slice_test.columns) - set(numeric_cols_test))
                        logger.warning(f"Trial {trial.number}: Dropping non-numeric columns before final .to_numpy() for test: {non_numeric_dropped_test}")
                    X_test_processed_np = df_slice_test[numeric_cols_test].astype(np.float32).to_numpy()
                else: 
                    X_test_processed_np = np.array([]).reshape(0, len(numeric_cols_train) if numeric_cols_train.any() else 0).astype(np.float32)

        else: # scaler was not initialized
            logger.warning(f"Trial {trial.number}: Scaler not initialized for method {current_normalization_method}, skipping scaling. Using unscaled (but possibly clipped) data.")
            df_slice_train_unscaled = df_to_scale_train[all_selected_features_ordered]
            numeric_cols_train_unscaled = df_slice_train_unscaled.select_dtypes(include=np.number).columns
            if len(numeric_cols_train_unscaled) < len(df_slice_train_unscaled.columns):
                logger.warning(f"Trial {trial.number}: Dropping non-numeric columns (unscaled path) for train: {list(set(df_slice_train_unscaled.columns) - set(numeric_cols_train_unscaled))}")
            X_train_processed_np = df_slice_train_unscaled[numeric_cols_train_unscaled].astype(np.float32).to_numpy()
            
            if not df_to_scale_val.empty:
                df_slice_val_unscaled = df_to_scale_val[all_selected_features_ordered]
                numeric_cols_val_unscaled = df_slice_val_unscaled.select_dtypes(include=np.number).columns
                if len(numeric_cols_val_unscaled) < len(df_slice_val_unscaled.columns):
                    logger.warning(f"Trial {trial.number}: Dropping non-numeric columns (unscaled path) for val: {list(set(df_slice_val_unscaled.columns) - set(numeric_cols_val_unscaled))}")
                X_val_processed_np = df_slice_val_unscaled[numeric_cols_val_unscaled].astype(np.float32).to_numpy()
            else:
                X_val_processed_np = np.array([]).reshape(0, len(numeric_cols_train_unscaled) if numeric_cols_train_unscaled.any() else 0).astype(np.float32)

            if not df_to_scale_test.empty:
                df_slice_test_unscaled = df_to_scale_test[all_selected_features_ordered]
                numeric_cols_test_unscaled = df_slice_test_unscaled.select_dtypes(include=np.number).columns
                if len(numeric_cols_test_unscaled) < len(df_slice_test_unscaled.columns):
                    logger.warning(f"Trial {trial.number}: Dropping non-numeric columns (unscaled path) for test: {list(set(df_slice_test_unscaled.columns) - set(numeric_cols_test_unscaled))}")
                X_test_processed_np = df_slice_test_unscaled[numeric_cols_test_unscaled].astype(np.float32).to_numpy()
            else:
                X_test_processed_np = np.array([]).reshape(0, len(numeric_cols_train_unscaled) if numeric_cols_train_unscaled.any() else 0).astype(np.float32)

    else: # Scaling is 'none' or no cols_to_scale
        logger.info(f"Trial {trial.number}: Scaling skipped (method: {current_normalization_method} or no columns to scale). Using unscaled (but possibly clipped) data.")
        df_slice_train_none = df_to_scale_train[all_selected_features_ordered]
        numeric_cols_train_none = df_slice_train_none.select_dtypes(include=np.number).columns
        if len(numeric_cols_train_none) < len(df_slice_train_none.columns):
            logger.warning(f"Trial {trial.number}: Dropping non-numeric columns (scaling='none' path) for train: {list(set(df_slice_train_none.columns) - set(numeric_cols_train_none))}")
        X_train_processed_np = df_slice_train_none[numeric_cols_train_none].astype(np.float32).to_numpy()
        
        if not df_to_scale_val.empty:
            df_slice_val_none = df_to_scale_val[all_selected_features_ordered]
            numeric_cols_val_none = df_slice_val_none.select_dtypes(include=np.number).columns
            if len(numeric_cols_val_none) < len(df_slice_val_none.columns):
                logger.warning(f"Trial {trial.number}: Dropping non-numeric columns (scaling='none' path) for val: {list(set(df_slice_val_none.columns) - set(numeric_cols_val_none))}")
            X_val_processed_np = df_slice_val_none[numeric_cols_val_none].astype(np.float32).to_numpy()
        else:
            X_val_processed_np = np.array([]).reshape(0, len(numeric_cols_train_none) if numeric_cols_train_none.any() else 0).astype(np.float32)

        if not df_to_scale_test.empty:
            df_slice_test_none = df_to_scale_test[all_selected_features_ordered]
            numeric_cols_test_none = df_slice_test_none.select_dtypes(include=np.number).columns
            if len(numeric_cols_test_none) < len(df_slice_test_none.columns):
                logger.warning(f"Trial {trial.number}: Dropping non-numeric columns (scaling='none' path) for test: {list(set(df_slice_test_none.columns) - set(numeric_cols_test_none))}")
            X_test_processed_np = df_slice_test_none[numeric_cols_test_none].astype(np.float32).to_numpy()
        else:
            X_test_processed_np = np.array([]).reshape(0, len(numeric_cols_train_none) if numeric_cols_train_none.any() else 0).astype(np.float32)


    # --- 5. Sequence Creation ---
    sequence_length = args.sequence_length
    
    # --- NEW: Create tf.data.Dataset for training using the generator ---
    if X_train_flat_df.empty or y_train_flat.empty:
        logger.error(f"Trial {trial.number}: X_train_flat_df or y_train_flat is empty before creating train_dataset. Pruning.")
        raise optuna.exceptions.TrialPruned("Empty training data for generator.")

    # Calculate steps_per_epoch for training data
    num_train_sequences_possible = len(X_train_processed_np) - sequence_length + 1
    if args.steps_per_epoch and args.steps_per_epoch > 0:
        steps_per_epoch_train = args.steps_per_epoch
        logger.info(f"Trial {trial.number}: Using user-defined steps_per_epoch for training: {steps_per_epoch_train}")
    else:
        if num_train_sequences_possible < 0: # Should not happen if X_train_processed_np is populated
            logger.error(f"Trial {trial.number}: num_train_sequences_possible is negative ({num_train_sequences_possible}). This indicates an issue with X_train_processed_np or sequence_length. Pruning.")
            raise optuna.exceptions.TrialPruned("Invalid num_train_sequences_possible.")
        steps_per_epoch_train = num_train_sequences_possible // cfg['batch_size']
        logger.info(f"Trial {trial.number}: Calculated steps_per_epoch for training: {steps_per_epoch_train} (num_train_seq: {num_train_sequences_possible}, batch: {cfg['batch_size']})")
    
    # Ensure steps_per_epoch is at least 1 if there's data and steps_per_epoch_train ended up as 0
    if steps_per_epoch_train == 0 and num_train_sequences_possible > 0:
        steps_per_epoch_train = 1
        logger.warning(f"Trial {trial.number}: steps_per_epoch_train was 0 after calculation, setting to 1 because num_train_sequences_possible is {num_train_sequences_possible}.")
    elif num_train_sequences_possible <= 0 and steps_per_epoch_train == 0: # Handles case of no sequences possible
        logger.warning(f"Trial {trial.number}: No training sequences possible ({num_train_sequences_possible}). steps_per_epoch_train remains 0. Training will likely be skipped by Keras or fail.")
        # Keras fit might handle steps_per_epoch=0 gracefully if dataset is also empty or yields nothing.

    train_generator = lambda: data_sequence_generator(
        X_train_processed_np, y_train_flat.to_numpy(), sequence_length, cfg['batch_size'], shuffle=True
    )
    train_dataset = tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, sequence_length, X_train_processed_np.shape[1]), dtype=X_train_processed_np.dtype),
            tf.TensorSpec(shape=(None, 1), dtype=y_train_flat.dtype) # Assuming target is int32 after cast
        )
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.repeat() # Ensure training generator can be re-iterated across epochs
    # No .repeat() here, handled by epochs in model.fit for TF >= 2.6 with steps_per_epoch
    logger.info(f"Trial {trial.number}: train_dataset created. Num possible train sequences: {len(X_train_processed_np) - sequence_length + 1}, steps_per_epoch_train: {steps_per_epoch_train}")

    # Validation dataset
    num_val_sequences_possible = len(X_val_processed_np) - sequence_length + 1
    steps_per_epoch_val = num_val_sequences_possible // args.batch_size_eval
    if steps_per_epoch_val == 0 and num_val_sequences_possible > 0: # Ensure at least one step if data exists
        steps_per_epoch_val = 1
        logger.warning(f"Trial {trial.number}: steps_per_epoch_val was 0, setting to 1.")

    val_generator = lambda: data_sequence_generator(
        X_val_processed_np, y_val_flat.to_numpy(), sequence_length, args.batch_size_eval, shuffle=False # No shuffle for val
    )
    val_dataset = tf.data.Dataset.from_generator(
        val_generator,
            output_signature=(
            tf.TensorSpec(shape=(None, sequence_length, X_val_processed_np.shape[1]), dtype=X_val_processed_np.dtype),
            tf.TensorSpec(shape=(None, 1), dtype=y_val_flat.dtype)
            )
        )
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    # Add .repeat() here because validation_steps is used in model.fit,
    # and F1EvalCallback also iterates over this dataset.
    # The .repeat() ensures it can be iterated multiple times.
    val_dataset = val_dataset.repeat() 
    logger.info(f"Trial {trial.number}: val_dataset created. Num possible val sequences: {num_val_sequences_possible}, steps_per_epoch_val: {steps_per_epoch_val}")

    # Test dataset (for evaluation within objective)
    num_test_sequences_possible = len(X_test_processed_np) - sequence_length + 1
    steps_per_epoch_test = num_test_sequences_possible // args.batch_size_eval
    if steps_per_epoch_test == 0 and num_test_sequences_possible > 0:
        steps_per_epoch_test = 1
        logger.warning(f"Trial {trial.number}: steps_per_epoch_test was 0, setting to 1.")

    test_generator = lambda: data_sequence_generator(
        X_test_processed_np, y_test_flat.to_numpy(), sequence_length, args.batch_size_eval, shuffle=False # No shuffle for test
    )
    test_dataset = tf.data.Dataset.from_generator(
        test_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, sequence_length, X_test_processed_np.shape[1]), dtype=X_test_processed_np.dtype),
            tf.TensorSpec(shape=(None, 1), dtype=y_test_flat.dtype)
        )
    )
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    # No .repeat() needed for test_dataset if used once with model.evaluate
    logger.info(f"Trial {trial.number}: test_dataset created. Num possible test sequences: {num_test_sequences_possible}, steps_per_epoch_test: {steps_per_epoch_test}")

    # --- Define data_shapes_for_config (after all datasets are defined) ---
    data_shapes_for_config = {
        'X_train_flat_df_shape': list(X_train_flat_df.shape) if hasattr(X_train_flat_df, 'shape') else None,
        'X_val_flat_df_shape': list(X_val_flat_df.shape) if hasattr(X_val_flat_df, 'shape') else None,
        'X_test_flat_df_shape': list(X_test_flat_df.shape) if hasattr(X_test_flat_df, 'shape') else None,
        'y_train_flat_shape': list(y_train_flat.shape) if hasattr(y_train_flat, 'shape') else None,
        'y_val_flat_shape': list(y_val_flat.shape) if hasattr(y_val_flat, 'shape') else None,
        'y_test_flat_shape': list(y_test_flat.shape) if hasattr(y_test_flat, 'shape') else None,
        
        'X_train_processed_np_shape': list(X_train_processed_np.shape) if hasattr(X_train_processed_np, 'shape') else None,
        'X_val_processed_np_shape': list(X_val_processed_np.shape) if hasattr(X_val_processed_np, 'shape') else None,
        'X_test_processed_np_shape': list(X_test_processed_np.shape) if hasattr(X_test_processed_np, 'shape') else None,
        
        'sequence_length': sequence_length,
        
        'num_train_sequences_possible': num_train_sequences_possible,
        'num_val_sequences_possible': num_val_sequences_possible,
        'num_test_sequences_possible': num_test_sequences_possible,

        'train_dataset_steps_per_epoch': steps_per_epoch_train,
        'val_dataset_steps_per_epoch': steps_per_epoch_val,
        'test_dataset_steps_per_epoch': steps_per_epoch_test,
        
        'num_features_for_model_input_layer': X_train_flat_df.shape[1] if hasattr(X_train_flat_df, 'shape') and X_train_flat_df.shape[1] is not None else None,
        'num_categorical_features_for_embedding': len(categorical_feature_specs) if categorical_feature_specs is not None else 0,
        'num_features_to_scale': len(cols_to_scale) if cols_to_scale is not None else 0
    }

    # Initialize sequence variables for callbacks and testing to ensure they exist
    X_val_seq_for_callbacks, y_val_seq_for_callbacks = np.array([]), np.array([])
    X_test_seq, y_test_seq = np.array([]), np.array([])

    # --- Apply --max-samples-for-np-eval before creating NumPy sequences ---
    X_val_processed_np_for_eval = X_val_processed_np
    y_val_flat_for_eval = y_val_flat.to_numpy() # Convert to numpy early for slicing
    X_test_processed_np_for_eval = X_test_processed_np
    y_test_flat_for_eval = y_test_flat.to_numpy()

    if args.max_samples_for_np_eval > 0:
        if len(X_val_processed_np_for_eval) > args.max_samples_for_np_eval:
            logger.info(f"Trial {trial.number}: Subsampling X_val_processed_np from {len(X_val_processed_np_for_eval)} to last {args.max_samples_for_np_eval} samples for NumPy sequence evaluation.")
            X_val_processed_np_for_eval = X_val_processed_np_for_eval[-args.max_samples_for_np_eval:]
            y_val_flat_for_eval = y_val_flat_for_eval[-args.max_samples_for_np_eval:]
        
        if len(X_test_processed_np_for_eval) > args.max_samples_for_np_eval:
            logger.info(f"Trial {trial.number}: Subsampling X_test_processed_np from {len(X_test_processed_np_for_eval)} to last {args.max_samples_for_np_eval} samples for NumPy sequence evaluation.")
            X_test_processed_np_for_eval = X_test_processed_np_for_eval[-args.max_samples_for_np_eval:]
            y_test_flat_for_eval = y_test_flat_for_eval[-args.max_samples_for_np_eval:]
    else:
        logger.info(f"Trial {trial.number}: --max-samples-for-np-eval is 0 or less. Using full validation/test sets for NumPy sequence creation. This might lead to OOM on large datasets.")

    # --- (Remove old create_sequences for val and test) ---
    logger.info(f"Trial {trial.number}: Using create_sequences for X_val_seq_for_callbacks and X_test_seq.")
    X_val_seq_for_callbacks, y_val_seq_for_callbacks = create_sequences(X_val_processed_np_for_eval, y_val_flat_for_eval, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test_processed_np_for_eval, y_test_flat_for_eval, sequence_length)
    logger.info(f"Trial {trial.number}: X_val_seq_for_callbacks shape: {X_val_seq_for_callbacks.shape}, y_val_seq_for_callbacks shape: {y_val_seq_for_callbacks.shape}")
    logger.info(f"Trial {trial.number}: X_test_seq shape: {X_test_seq.shape}, y_test_seq shape: {y_test_seq.shape}")
    
    # --- 7. Build Model ---
    # n_features_total is the number of features in the flat data before sequencing by the generator
    n_features_total = X_train_flat_df.shape[1] 
    logger.info(f"Trial {trial.number}: Total features for model input (from X_train_flat_df.shape[1]): {n_features_total}")
    if n_features_total == 0:
        logger.error(f"Trial {trial.number}: No features for model input (n_features_total is 0 from X_train_flat_df). Pruning.")
        raise optuna.exceptions.TrialPruned("No features for model input from X_train_flat_df.")
    if n_features_total != len(all_selected_features_ordered):
        logger.warning(f"Trial {trial.number}: Mismatch between n_features_total from X_train_flat_df.shape[1] ({n_features_total}) and len(all_selected_features_ordered) ({len(all_selected_features_ordered)}). This might indicate issues in data processing or list management.")

    # +++ Calculate pos/neg counts for bias initializer +++
    # Use y_train_flat for bias calculation as it reflects the distribution before sequencing
    pos_count = np.sum(y_train_flat == 1)
    neg_count = np.sum(y_train_flat == 0)
    logger.info(f"Trial {trial.number}: Training flat data class counts for bias initializer - Pos: {pos_count}, Neg: {neg_count}")
    # Avoid division by zero if one class is absent, though this should be rare with enough data
    # output_bias_init_val = np.log(pos_count / neg_count) if pos_count > 0 and neg_count > 0 else 0.0 # REMOVED
    output_bias_init_val = None # Set to None to remove output bias initializer
    logger.info(f"Trial {trial.number}: Output bias initializer REMOVED (set to None).")
    # +++ End bias initializer calculation +++

    # +++ Define Optuna Hyperparameters for Model Config (cfg dictionary) +++
    # This section consolidates hyperparameters from Optuna's `hp` suggestion
    # into the `cfg` dictionary format expected by `build_model` and other parts.
    
    # Determine ff_dim for Transformer based on d_model_transformer_mha and a factor
    # d_model_transformer_mha is num_heads * head_size
    # MOVED cfg DEFINITION EARLIER
    # d_model_transformer_mha = hp.get('num_heads', args.num_heads) * hp.get('head_size', args.head_size)
    # transformer_ff_dim = int(d_model_transformer_mha * hp.get('ff_dim_factor', args.ff_dim_factor))

    # Define CNN layer configurations based on Optuna suggestions for a single stack.
    # For this version, we'll define a fixed number of CNN layers (e.g., 2 or 3) 
    # and tune their parameters. Let's assume 2 CNN layers for this example.
    # In a more complex setup, the number of layers could also be an Optuna parameter.
    # MOVED cfg DEFINITION EARLIER
    # cnn_layer_configs_from_hp = [] 
    # Example for 2 CNN layers, more can be added similarly
    # num_cnn_layers_to_tune = 2 # Fixed for now, can be an HP later
    # for i in range(1, num_cnn_layers_to_tune + 1):
    #     cnn_layer_configs_from_hp.append({
    #         'filters': hp.get(f'cnn_filters_{i}', 64), # Default if not in hp (should be)
    #         'kernel': hp.get(f'cnn_kernel_size_{i}', 5), # Default if not in hp
    #         'dilation': hp.get(f'cnn_dilation_rate_{i}', 1) # Default, can add to Optuna
    #     })

    # MOVED cfg DEFINITION EARLIER
    # cfg = {
    #     'cnn_layer_configs': cnn_layer_configs_from_hp,
    #     'cnn_pool_size': hp.get('cnn_pool_size', 2), # Default, can add to Optuna
    #     'cnn_pool_strides': hp.get('cnn_pool_strides', 2), # Default, can add to Optuna
    #     'cnn_dropout_rate': hp.get('cnn_dropout_rate', 0.1), # Default, can add to Optuna
    #     'cnn_activation': args.cnn_activation, # Usually fixed for a study, e.g., 'gelu'
        
    #     'num_transformer_blocks': hp.get('num_transformer_blocks', args.num_transformer_blocks),
    #     'num_heads': hp.get('num_heads', args.num_heads),
    #     'head_size': hp.get('head_size', args.head_size),
    #     'ff_dim': transformer_ff_dim, # Calculated above
    #     'transformer_dropout_rate': hp.get('dropout_rate', args.dropout_rate), # General dropout

    #     'mlp_units': hp.get('mlp_units', [int(u) for u in args.mlp_units]), # Ensure MLP units are integers
    #     'mlp_dropout_rate': hp.get('mlp_dropout_rate', args.mlp_dropout_rate), # Tied to general dropout or separate HP

    #     'l2_strength': args.l2_strength, # Usually fixed for a study
    #     'ffn_activation': 'gelu', # Common choice, can be HP

    #     'learning_rate': hp.get('learning_rate', args.learning_rate),
    #     'weight_decay': hp.get('weight_decay', args.weight_decay),
    #     'batch_size': hp.get('batch_size', args.batch_size),

    #     'focal_alpha': hp.get('focal_alpha', args.focal_alpha),
    #     'focal_gamma': hp.get('focal_gamma', args.focal_gamma),
        
    #     # Store trial number in cfg for callbacks if needed for paths
    #     'trial_number_for_paths': trial.number,
        
    #     # Instance Normalization Hyperparameters
    #     'use_instance_norm': args.use_instance_norm, # For now, controlled by CLI args
    #     'instance_norm_affine': trial.suggest_categorical('instance_norm_affine_optuna', [True, False]) # OPTUNA TUNABLE
    # }
    # logger.info(f"Trial {trial.number}: Hyperparameters (cfg) for this trial: {json.dumps(cfg, indent=2, default=str)}")

    # +++ Define Categorical Feature Specifications for Embedding Layers +++
    logger.info(f"Trial {trial.number}: Categorical feature specs for embeddings: {categorical_feature_specs}")

    # --- A: Save Full Trial Configuration --- (Part 2: Compile and Save)
    if trial_path and args.save_results:
        full_trial_config_to_save = {
            'trial_number': trial.number,
            'cli_args': {k: v for k, v in vars(args).items() if k not in ['wandb_api_key']}, # Exclude sensitive args if any
            'optuna_hyperparameters_chosen (hp)': hp,
            'derived_model_config (cfg)': cfg,
            'data_shapes': data_shapes_for_config,
            'input_parquet_file_details': {
                'path': args.feature_parquet,
                'readme_notes_on_log_volumes': "Assumes consolidated_features_targets_all.parquet: has log-vol for time bars, raw-vol for dollar bars."
            }
        }
        config_save_path = trial_path / f"trial_{trial.number:03d}_full_config.json"
        try:
            with open(config_save_path, 'w') as f_config:
                json.dump(full_trial_config_to_save, f_config, indent=4, default=str) # Use default=str for non-serializable like Path
            logger.info(f"Trial {trial.number}: Saved full trial configuration to {config_save_path}")
        except Exception as e_config_save:
            logger.error(f"Trial {trial.number}: Error saving full trial configuration: {e_config_save}")
    # --- End A Part 2 ---

    tf.keras.backend.clear_session()
    model = build_model( 
        sequence_length=args.sequence_length,
        n_features_total=n_features_total, # This is the count *before* embeddings expand dimensions
        all_feature_names=all_selected_features_ordered, # NEW: Pass the ordered list of all feature names
        categorical_feature_specs=categorical_feature_specs, # NEW: Pass the specs for embedding layers
        # --- Instance Norm ---
        use_instance_norm=cfg['use_instance_norm'],
        instance_norm_affine=cfg['instance_norm_affine'], # Use the Optuna-tuned or default value
        # --- CNN ---
        cnn_layer_configs=cfg['cnn_layer_configs'],
        cnn_pool_size=cfg['cnn_pool_size'],
        cnn_pool_strides=cfg['cnn_pool_strides'],
        cnn_dropout_rate=cfg['cnn_dropout_rate'],
        cnn_activation=cfg['cnn_activation'],
        cnn_use_bias=(not args.cnn_no_bias),
        num_transformer_blocks=cfg['num_transformer_blocks'],
        num_heads=cfg['num_heads'],
        head_size=cfg['head_size'],
        ff_dim=cfg['transformer_ff_dim'], # Corrected to use 'transformer_ff_dim' from cfg
        mlp_units=cfg['mlp_units'],
        transformer_dropout_rate=cfg['transformer_dropout_rate'],
        mlp_dropout_rate=cfg['mlp_dropout_rate'],
        l2_strength=cfg['l2_strength'],
        ffn_activation=cfg['ffn_activation'], # Pass ffn_activation
        output_bias_init_value=output_bias_init_val # This argument must be last if it has a default
    )

    # --- Learning Rate Schedule Setup ---
    warmup_epochs_val = 3 # Consistent with get_callbacks
    
    # steps_per_epoch_train is already calculated and validated (>=1 or pruned) during train_dataset setup.
    # We just need to assign it to the 'steps_per_epoch' variable used by the CosineDecay logic.
    steps_per_epoch = steps_per_epoch_train 

    total_epochs_for_decay = args.epochs - warmup_epochs_val
    if total_epochs_for_decay <= 0:
        logger.warning(f"Trial {trial.number}: Total epochs for decay ({total_epochs_for_decay}) is not positive. CosineDecay might not work as expected or LR will remain constant after warmup. Ensure args.epochs > warmup_epochs_val ({warmup_epochs_val}).")
        # If decay epochs is not positive, decay_steps might be zero or negative.
        # Defaulting to a minimal decay_steps to avoid tf.errors.InvalidArgumentError if decay_steps must be positive.
        # Or, one might choose to not use CosineDecay here and just stick with the warmup_lr.
        # For now, let CosineDecay handle it, it might just keep LR constant at initial_learning_rate.
        decay_steps = steps_per_epoch # Minimal steps if no decay epochs
    else:
        decay_steps = steps_per_epoch * total_epochs_for_decay
    
    logger.info(f"Trial {trial.number}: LR Schedule: Warmup epochs: {warmup_epochs_val}, Peak LR: {cfg['learning_rate']:.2e}")
    logger.info(f"Trial {trial.number}: LR Schedule: Cosine decay for {total_epochs_for_decay} epochs over {decay_steps} steps after warmup.")

    cosine_decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=cfg["learning_rate"], # Peak LR reached after warmup
        decay_steps=max(1, decay_steps), # Ensure decay_steps is at least 1
        alpha=0.01 # Decay to 1% of the peak LR (alpha=0.0 means decay to zero)
    )
    # The LinearWarmUp callback will directly manipulate the optimizer's LR value for the warmup phase.
    # The AdamW optimizer is initialized with the cosine_decay_schedule. After warmup, the LR value set by
    # LinearWarmUp will be cfg["learning_rate"], and then the cosine_decay_schedule logic effectively continues from there.
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=cosine_decay_schedule, # Use the cosine decay schedule here
        weight_decay=cfg["weight_decay"],
        clipnorm=1.0  # Add gradient clipping
    )
    # +++ FOR DEBUGGING: Use BCE for trial 0 if it's the forced HP trial +++
    # if trial.number == 0: # REMOVE DEBUG CODE
    #     logger.warning("DEBUG: Forcing BinaryCrossentropy loss for Trial 0 for NaN debugging.")
    #     loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
    # else:
    #     loss_fn = create_focal_loss(gamma=cfg["focal_gamma"], alpha=cfg["focal_alpha"], label_smoothing=args.label_smoothing) if args.use_focal_loss else tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
    
    # All trials will now use Focal Loss with Optuna-tuned alpha and gamma (if --use-focal-loss is on, which is default)
    loss_fn = create_focal_loss(gamma=cfg["focal_gamma"], alpha=cfg["focal_alpha"], label_smoothing=args.label_smoothing) if args.use_focal_loss else tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)

    metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc'), SafeF1Score(name='f1_score')]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    logger.info(f"Trial {trial.number}: Model built and compiled.")

    if args.save_results and trial_path and trial.number < 3: # Save for first few trials
        try:
            with open(trial_path / "model_summary.txt", 'w') as f: model.summary(print_fn=lambda x: f.write(x + '\n'))
            if not args.skip_plots: tf.keras.utils.plot_model(model, to_file=trial_path / "model_plot.png", show_shapes=True, expand_nested=True)
        except Exception as e: logger.warning(f"Trial {trial.number}: Error saving model summary/plot: {e}")

    # --- 7. Training Setup (Class Weights and Callbacks) ---
    class_weights = None
    # o3 recommendation: Disable class weights when tuning Focal Loss.
    # We will ensure class_weights is None for Optuna runs, regardless of CLI,
    # as Focal Loss (if used) is handling re-balancing.
    # If Focal Loss is not used, class weights could still be considered, but o3 says "disable for this round".
    # logger.info(f"Trial {trial.number}: Class weights are DISABLED for this Optuna run (o3 recommendation). CLI arg --use-class-weights is ignored.")
    # Original logic for class_weights:
    # Updated class weight logic based on o3's advice regarding focal_alpha
    class_weights = None # Default to no class weights
    if abs(cfg['focal_alpha'] - 0.5) < 1e-3: # o3: Only use class_weights if alpha is very near 0.5
        logger.info(f"Trial {trial.number}: Focal alpha ({cfg['focal_alpha']:.3f}) is ~0.5. Class weight application will respect --use-class-weights CLI arg.")
        if args.use_class_weights:
            # Class weights should be computed on y_train_flat (before sequencing for generator)
            if y_train_flat.size > 0 : 
                unique_classes_flat = np.unique(y_train_flat.astype(int)) # y_train_flat is a Series
                if len(unique_classes_flat) > 1: 
                     computed_weights = sk_class_weight.compute_class_weight(
                         'balanced', 
                         classes=unique_classes_flat, 
                         y=y_train_flat.astype(int) # Use y_train_flat here
                     )
                     class_weights = dict(zip(unique_classes_flat, computed_weights))
                     logger.info(f"Trial {trial.number}: Using balanced class weights (from y_train_flat): {class_weights}")
                else: 
                     logger.warning(f"Trial {trial.number}: Only one class found in y_train_flat (alpha ~0.5). Cannot compute balanced class weights. Proceeding without class weights.")
            else:
                logger.warning(f"Trial {trial.number}: y_train_flat is empty (alpha ~0.5). Cannot compute class weights. Proceeding without class weights.")
        else: 
            logger.info(f"Trial {trial.number}: Class weights explicitly disabled by --no-use-class-weights CLI argument (alpha ~0.5).")
    else: 
        logger.info(f"Trial {trial.number}: Focal alpha ({cfg['focal_alpha']:.3f}) is not ~0.5. Class weights will be DISABLED as per o3's recommendation to avoid double-counting (CLI arg --use-class-weights ignored in this case).")

    callbacks = get_callbacks(
        args=args,
        run_dir=trial_path,
        val_dataset=val_dataset, # Pass val_dataset
        batch_size=args.batch_size_eval, # This batch_size is for the old F1 callback, might be redundant
        hp=cfg,
        num_val_sequences_possible=num_val_sequences_possible,
        val_steps_for_callback=steps_per_epoch_val # Pass steps_per_epoch_val
    )
    
    # --- 8. Model Training ---
    # y_train_seq_reshaped and y_val_seq_reshaped are NOT used with tf.data.Dataset pipeline
    # validation_data_tuple also not used in the same way.
    
    history = None
    best_model_checkpoint_path = trial_path / 'best_auc_model.keras' # Define path to best model
    try:
        logger.info(f"Trial {trial.number}: Starting model.fit() with tf.data.Dataset - Epochs: {args.epochs}, Batch: {cfg['batch_size']}, Steps per Epoch Train: {steps_per_epoch_train}")
        
        fit_validation_data_arg = None
        fit_validation_steps_arg = None

        if val_dataset:
            if steps_per_epoch_val is not None and steps_per_epoch_val > 0:
                logger.info(f"Trial {trial.number}: Validation will use tf.data.Dataset with {steps_per_epoch_val} steps.")
                fit_validation_data_arg = val_dataset
                fit_validation_steps_arg = steps_per_epoch_val
            # elif X_val_seq_for_callbacks.size > 0 and y_val_seq_for_callbacks.size > 0: # Fallback to NumPy if val_dataset steps are 0 but callback data exists
            #     logger.warning(f"Trial {trial.number}: val_dataset created but steps_per_epoch_val is {steps_per_epoch_val}. "
            #                    f"Falling back to NumPy validation data (X_val_seq_for_callbacks) for model.fit() if available and non-empty.")
            #     fit_validation_data_arg = (X_val_seq_for_callbacks, y_val_seq_for_callbacks.reshape(-1,1))
            #     # validation_steps is not needed if validation_data is NumPy array
            else: # val_dataset exists, but steps are 0, and no NumPy fallback
                logger.warning(f"Trial {trial.number}: val_dataset exists but steps_per_epoch_val is {steps_per_epoch_val}, and no NumPy fallback. Validation might not run as expected in model.fit().")
                if steps_per_epoch_val == 0:
                    fit_validation_data_arg = None # Explicitly no validation for fit if steps are 0
        # elif X_val_seq_for_callbacks.size > 0 and y_val_seq_for_callbacks.size > 0: # No val_dataset, but NumPy data for callbacks exists
        #     logger.info(f"Trial {trial.number}: No val_dataset created. Using NumPy validation data (X_val_seq_for_callbacks) for model.fit().")
        #     fit_validation_data_arg = (X_val_seq_for_callbacks, y_val_seq_for_callbacks.reshape(-1,1))
        else: # No val_dataset and no NumPy validation data
            logger.info(f"Trial {trial.number}: No validation dataset for model.fit().")

        history = model.fit(
            train_dataset, 
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch_train, 
            validation_data=fit_validation_data_arg,
            validation_steps=fit_validation_steps_arg,
            callbacks=callbacks,
            class_weight=class_weights, 
            verbose=args.keras_verbose
        )
        logger.info(f"Trial {trial.number}: Model training finished normally.")
    except KeyboardInterrupt:
        logger.warning(f"Trial {trial.number}: KeyboardInterrupt detected during model.fit(). Training was stopped prematurely.")
        logger.info(f"Attempting to load weights from best checkpoint: {best_model_checkpoint_path}")
        if best_model_checkpoint_path.exists():
            try:
                model.load_weights(str(best_model_checkpoint_path))
                logger.info(f"Successfully loaded weights from {best_model_checkpoint_path}.")
            except Exception as e_load:
                logger.error(f"Error loading weights from {best_model_checkpoint_path}: {e_load}. Proceeding with current model weights.")
        else:
            logger.warning(f"Best model checkpoint {best_model_checkpoint_path} not found. Proceeding with current model weights.")
    except RuntimeError as e:
        if "nan loss" in str(e).lower():
            logger.error(f"Trial {trial.number}: NaN loss during training. Pruning. Error: {e}")
            raise optuna.exceptions.TrialPruned("NaN loss during training.")
        logger.error(f"Trial {trial.number}: Keras RuntimeError: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned(f"Keras fit RuntimeError: {e}")
    except Exception as e:
        logger.error(f"Trial {trial.number}: Unexpected error during fit: {e}. Pruning.")
        raise optuna.exceptions.TrialPruned(f"Unexpected fit error: {e}")

    # --- 9. Post-Training Evaluation & Optuna Metric Return ---
    
    # --- 9a. Test Set Evaluation (Uncalibrated and Calibrated) ---
    test_metrics_results = {}
    if X_test_seq.size > 0 and y_test_seq.size > 0:
        logger.info(f"Trial {trial.number}: Evaluating on test set...")
        y_pred_test_proba_uncalibrated = model.predict(X_test_seq, batch_size=args.batch_size_eval, verbose=0).flatten()
        
        # Store uncalibrated metrics
        test_metrics_results['uncalibrated'] = {}
        for thresh_pct in [0.5, 0.55, 0.6, 0.65, 0.7]: # Evaluate at multiple thresholds
            y_pred_test_binary_uncalibrated = (y_pred_test_proba_uncalibrated >= thresh_pct).astype(int)
            report_uncal = classification_report(y_test_seq, y_pred_test_binary_uncalibrated, output_dict=True, zero_division=0)
            test_metrics_results['uncalibrated'][f'report_thresh_{thresh_pct:.2f}'] = report_uncal
            test_metrics_results['uncalibrated'][f'accuracy_thresh_{thresh_pct:.2f}'] = accuracy_score(y_test_seq, y_pred_test_binary_uncalibrated)
            test_metrics_results['uncalibrated'][f'roc_auc_thresh_{thresh_pct:.2f}'] = roc_auc_score(y_test_seq, y_pred_test_proba_uncalibrated) # AUC is threshold-independent

        # --- Isotonic Calibration ---
        calibrator = None
        if X_val_seq_for_callbacks.size > 0 and y_val_seq_for_callbacks.size > 0:
            logger.info(f"Trial {trial.number}: Fitting Isotonic Calibrator on validation data...")
            y_pred_val_proba_for_calibration = model.predict(X_val_seq_for_callbacks, batch_size=args.batch_size_eval, verbose=0).flatten()
            
            # --- B: Enhanced Isotonic Calibration Logging ---
            logger.info(f"Trial {trial.number}: Shapes for Isotonic Calibrator fit - y_pred_val_proba: {y_pred_val_proba_for_calibration.shape}, y_val_seq_for_callbacks: {y_val_seq_for_callbacks.shape}")
            # --- End B ---

            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            try:
                calibrator.fit(y_pred_val_proba_for_calibration, y_val_seq_for_callbacks.astype(int)) # Ensure y_val_seq_for_callbacks is int
                logger.info(f"Trial {trial.number}: Isotonic Calibrator fitted.")
                if trial_path and args.save_results:
                    calibrator_save_path = trial_path / f"isotonic_calibrator_trial_{trial.number}.joblib"
                    joblib.dump(calibrator, calibrator_save_path)
                    logger.info(f"Trial {trial.number}: Saved isotonic calibrator to {calibrator_save_path}")
            except Exception as e_calib_fit:
                 logger.error(f"Trial {trial.number}: Error fitting Isotonic Calibrator: {e_calib_fit}. Skipping calibration for test set.", exc_info=True)
                 calibrator = None # Ensure calibrator is None if fitting failed
        else:
            logger.warning(f"Trial {trial.number}: Validation data empty, skipping Isotonic Calibrator training.")

        if calibrator:
            logger.info(f"Trial {trial.number}: Applying Isotonic Calibrator to test predictions...")
            y_pred_test_proba_calibrated = calibrator.predict(y_pred_test_proba_uncalibrated)
            
            test_metrics_results['calibrated'] = {}
            for thresh_pct in [0.5, 0.55, 0.6, 0.65, 0.7]: # Evaluate at multiple thresholds
                y_pred_test_binary_calibrated = (y_pred_test_proba_calibrated >= thresh_pct).astype(int)
                report_cal = classification_report(y_test_seq, y_pred_test_binary_calibrated, output_dict=True, zero_division=0)
                test_metrics_results['calibrated'][f'report_thresh_{thresh_pct:.2f}'] = report_cal
                test_metrics_results['calibrated'][f'accuracy_thresh_{thresh_pct:.2f}'] = accuracy_score(y_test_seq, y_pred_test_binary_calibrated)
                test_metrics_results['calibrated'][f'roc_auc_thresh_{thresh_pct:.2f}'] = roc_auc_score(y_test_seq, y_pred_test_proba_calibrated) # AUC is threshold-independent
            # else: # This was 'else:' - removed as it caused issues, logic flow should handle this.
        else:
            logger.warning(f"Trial {trial.number}: Calibrator not available. Skipping calibrated test metrics.")

        if trial_path and args.save_results and test_metrics_results:
            test_metrics_path = trial_path / f"test_metrics_eval_trial_{trial.number}.json"
            with open(test_metrics_path, 'w') as f:
                json.dump(test_metrics_results, f, indent=4, default=str)
            logger.info(f"Trial {trial.number}: Saved test set evaluation metrics to {test_metrics_path}")
        # else: # This was 'else:' - removed, logic flow handles this.
    else:
        logger.warning(f"Trial {trial.number}: Test set empty. Skipping test set evaluation and calibration.")

    # --- 9b. Optuna Metric Return ---
    optuna_metric_to_return = 0.0
    if history and hasattr(history, 'history') and history.history:
        # Prefer best_val_f1 logged by F1EvalCallback, accessed via ManualHistorySaver
        if trial_path: # Corrected variable: Check if artifacts are being saved (trial_path would exist)
            trial_num_str_load = f"trial_{trial.number:03d}"
            # The ManualHistorySaver callback saves to history_XXX.json directly in trial_path
            history_json_path = trial_path / f'history_{trial_num_str_load}.json' # Corrected variable
            if history_json_path.exists():
                try:
                    with open(history_json_path, 'r') as f: hist_data = json.load(f)
                    # F1EvalCallback logs 'best_val_f1' as a scalar that represents the best F1 *so far* in training.
                    # We want the final best 'best_val_f1' from the training run.
                    # The ManualHistorySaver stores 'best_val_f1' as a list if it's logged by F1EvalCallback.
                    if 'best_val_f1' in hist_data and isinstance(hist_data['best_val_f1'], list) and hist_data['best_val_f1']:
                        valid_f1s = [f for f in hist_data['best_val_f1'] if f is not None and not np.isnan(f)]
                        if valid_f1s: optuna_metric_to_return = float(max(valid_f1s))
                    elif 'val_auc' in hist_data and isinstance(hist_data['val_auc'], list) and hist_data['val_auc']: # Fallback if F1 not list
                        valid_aucs = [auc for auc in hist_data['val_auc'] if auc is not None and not np.isnan(auc)]
                        if valid_aucs: optuna_metric_to_return = float(max(valid_aucs))
                except Exception as e_hist_load:
                    logger.warning(f"Trial {trial.number}: Error loading/parsing F1/AUC from history.json: {e_hist_load}")
        
        # Fallback to val_auc from Keras history if metric not found or error from JSON
        if optuna_metric_to_return == 0.0 and 'val_auc' in history.history and history.history['val_auc']:
            valid_aucs = [auc for auc in history.history['val_auc'] if auc is not None and not np.isnan(auc)]
            if valid_aucs: 
                optuna_metric_to_return = float(max(valid_aucs))
                logger.info(f"Trial {trial.number}: Using max val_auc from Keras history ({optuna_metric_to_return:.4f}) as Optuna metric (F1/AUC from JSON not found/error).")
            else: 
                logger.warning(f"Trial {trial.number}: val_auc in Keras history was empty or all None/NaN.")
        elif optuna_metric_to_return != 0.0:
             logger.info(f"Trial {trial.number}: Using metric from history.json ({optuna_metric_to_return:.4f}) as Optuna metric.")
        else:
            logger.error(f"Trial {trial.number}: No suitable metric (best_val_f1 or val_auc) found. Returning 0.0.")
    else:
        logger.error(f"Trial {trial.number}: Keras history object not available or empty. Returning 0.0.")

    logger.info(f"--- Trial {trial.number} Finished. Returning Optuna metric: {optuna_metric_to_return:.6f} ---")
    return optuna_metric_to_return

def get_next_run_timestamp_dir_name(parent_dir: Path) -> str:
    """
    Scans parent_dir for existing run folders (e.g., '001_timestamp', '002_timestamp')
    and returns the name for the next sequential run folder, including a new timestamp.
    """
    pattern = re.compile(r"^(\\d{3})_.*") # Match 3 digits at the start, followed by underscore
    max_run_num = 0
    if parent_dir.exists() and parent_dir.is_dir():
        for name in os.listdir(parent_dir):
            p_path = parent_dir / name
            if p_path.is_dir(): # Only consider directories
                match = pattern.match(name)
                if match:
                    try:
                        num = int(match.group(1))
                        if num > max_run_num:
                            max_run_num = num
                    except ValueError:
                        pass 
    next_run_num = max_run_num + 1
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{next_run_num:03d}_{timestamp}"
    logger.info(f"Determined next run directory name: {folder_name} in {parent_dir}")
    return folder_name

def main():
    """Main function to setup and run Optuna study."""
    args = parse_args()

    # args.output_dir (e.g., "minotaur_5_31_outputs/") is the base directory where all specific run folders will reside.
    optuna_runs_parent_path = Path(args.output_dir)
    optuna_runs_parent_path.mkdir(parents=True, exist_ok=True)

    # Get the name for the next numbered and timestamped run folder (e.g., "001_20250601_103000")
    current_run_dir_name = get_next_run_timestamp_dir_name(optuna_runs_parent_path)
    
    # This is the specific directory for *this* execution of the script.
    # e.g., minotaur_5_31_outputs/001_20250601_103000/
    current_run_artifacts_dir = optuna_runs_parent_path / current_run_dir_name
    
    try:
        logger.info(f"Attempting to create directory for this script execution: {current_run_artifacts_dir}")
        current_run_artifacts_dir.mkdir(parents=True, exist_ok=False) # exist_ok=False to ensure it's new
        logger.info(f"Successfully created script execution artifacts directory: {current_run_artifacts_dir}")
    except FileExistsError:
        logger.error(f"Directory {current_run_artifacts_dir} unexpectedly already exists. "
                       f"This indicates an issue with run numbering or a race condition. Exiting to prevent overwrite.")
        # It's safer to exit if a uniquely named folder already exists.
        return # Or raise an error

    logger.info(f"All outputs for this script execution (DB and trial artifacts) will be in: {current_run_artifacts_dir}")

    # Construct the Optuna database path using the filename from args, placing it inside current_run_artifacts_dir
    actual_storage_path = f"sqlite:///{current_run_artifacts_dir / args.optuna_db_filename}" 
    
    # --- D: Prepare WandbCallback for Optuna --- (Part 1: Instantiate)
    wandb_optuna_callback = None
    # if args.use_wandb: # TEMP COMMENT OUT FOR SMOKE TEST
    #     try:
    #         import wandb # Ensure wandb is imported here too for this block
    #         # WandbCallback can be added to study.optimize if trial-specific logging to wandb is desired
            
    #         # Define metrics to be reported to WandB by the Optuna callback
    #         # This should usually be the same metric Optuna is optimizing (e.g., 'value' which is val_auc or best_val_f1)
    #         # If you want to log more things from Keras history directly to WandB for each trial,
    #         # you'd typically use the Keras WandbCallback within model.fit().
    #         # The Optuna WandbCallback primarily logs the trial's objective value and hyperparameters.
    #         wandb_optuna_callback = WandbCallback(
    #             metric_name=f"optuna_objective_{args.optuna_direction}", # e.g., optuna_objective_maximize
    #             wandb_kwargs={
    #                 "project": args.wandb_project,
    #                 "entity": args.wandb_entity,
    #                 "reinit": True, # Reinit wandb run for each Optuna trial if desired
    #                 "group": args.optuna_study_name, # Group trials by study name
    #                 # Name for each trial run in WandB can be auto-generated or set using {trial.number}
    #                 # "name": f"trial-{trial.number:03d}-{args.optuna_study_name}" #This won't work here, trial is not defined
    #                                                                         # Instead, WandbCallback usually names it internally
    #                                                                         # or you can configure it to use trial.number if it supports it.
    #                                                                         # For now, rely on default or group naming.
    #             }
    #         )
    #         # The WandbCallback will be passed to study.optimize() later
    #         logger.info("Optuna WandbCallback prepared. It will log each trial's objective and HPs to WandB.")
    #         logger.info(f"WandB Project: {args.wandb_project}, Entity: {args.wandb_entity}, Group: {args.optuna_study_name}")

    #         # Initialize a main WandB run for the overall Optuna study overview if not already done
    #         # This is separate from the per-trial runs that WandbCallback might create.
    #         if wandb.run is None: # Check if a run is already active
    #             wandb.init(
    #                 project=args.wandb_project,
    #                 entity=args.wandb_entity, 
    #                 config=vars(args), # Log CLI args for the main study run
    #                 reinit=False, # Do not reinit if already initialized by user earlier
    #                 name=f"optuna-study-{args.optuna_study_name}-PID{os.getpid()}",
    #                 group=args.optuna_study_name # Group this overview run with trial runs
    #             )
    #             logger.info("Main Weights & Biases run initialized for Optuna study overview.")
    #         else:
    #             logger.info("Main Weights & Biases run was already active.")

    #     except ImportError:
    #         logger.warning("wandb not installed, skipping WandB integration for Optuna callback. Run 'pip install wandb'")
    #         args.use_wandb = False # Disable further WandB attempts if import fails
    #     except Exception as e:
    #         logger.error(f"Error preparing or initializing WandB for Optuna callback: {e}")
    #         args.use_wandb = False # Disable further WandB attempts
    # --- End D Part 1 ---

    logger.info(f"Optuna study name: {args.optuna_study_name}")
    logger.info(f"Optuna direction: {args.optuna_direction}")
    logger.info(f"Optuna storage DB will be at: {actual_storage_path}")

    if args.no_optuna:
        logger.info("Optuna is disabled (--no-optuna). Running a single trial with fixed hyperparameters.")
        args.n_trials = 1 # Override n_trials for a single run

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        direction=args.optuna_direction,
        storage=actual_storage_path, 
        load_if_exists=True               
    )

    # --- D: Pass WandbCallback to Optuna study.optimize --- (Part 2)
    optuna_callbacks_list = []
    # if args.use_wandb and wandb_optuna_callback: # TEMP COMMENT OUT FOR SMOKE TEST
    #     optuna_callbacks_list.append(wandb_optuna_callback)
    #     logger.info("WandbCallback will be used for Optuna study.optimize().")
    # --- End D Part 2 ---

    try:
        # Pass current_run_artifacts_dir to the objective function so it knows where to save trial artifacts
        study.optimize(lambda trial: objective(trial, args, current_run_artifacts_dir), 
                       n_trials=args.n_trials,
                       callbacks=optuna_callbacks_list if optuna_callbacks_list else None) # Pass callbacks list
    except KeyboardInterrupt:
        logger.warning("Optuna study interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during the Optuna study: {e}", exc_info=True)

    logger.info("Optuna study finished.")
    logger.info(f"Number of finished trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        logger.info("Best trial:")
        logger.info(f"  Value (metric): {best_trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
    except ValueError: # Handles case where study has no completed trials (e.g. all pruned)
        logger.warning("No best trial found (e.g., study interrupted, all trials failed/pruned, or no trials run).")
    except AttributeError: # Handles case where study might be empty if --no-optuna and study object isn't fully populated for 'best_trial'
        if args.no_optuna and len(study.trials) == 1:
            logger.info(f"Single trial run completed. Metric: {study.trials[0].value:.4f}")
            logger.info(f"  Params used (fixed):")
            # Log the fixed params used in the 'hp' dict for the single trial
            # This requires passing them or re-constructing them here.
            # For now, this log might be less detailed for single run.
            # The trial_XXX_full_config.json will have the exact HPs.
        else:
            logger.warning("AttributeError when accessing best_trial. Study might be empty or in an unexpected state.")

    
    logger.info(f"Optuna study artifacts (including DB) are in: {current_run_artifacts_dir}")
    logger.info("Optuna integration main script finished.")

if __name__ == "__main__":
    main() 