from __future__ import annotations

# Add parent directory to path.
from pathlib import Path
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # https://stackoverflow.com/a/64438413
fdir = Path(__file__).parent.resolve() # Directory of current file.
path = fdir/'..'
if path not in sys.path:
    sys.path.append(str(path))

# Load config file if available.
import yaml
config_path = fdir/'config.yml'
if config_path.exists():
    config = yaml.safe_load(open(config_path))

# Process configs.
for key in config['roots']:
    # Convert to path.
    config['roots'][key] = Path(config['roots'][key]).expanduser()
    # Create path if necessary.
    config['roots'][key].mkdir(parents=True, exist_ok=True)

# Complete imports.
import argparse
import inspect
import logging
import makassar_ml as ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
from tqdm.keras import TqdmCallback

# Set random seeds.
SEED = 0
tf.random.set_seed(SEED) # Only this works on ARC (since tensorflow==2.4).

# Setup logging (useful for ARC systems).
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Must be lowest of all handlers listed below.
while logger.hasHandlers(): logger.removeHandler(logger.handlers[0]) # Clear all existing handlers.

# Custom log formatting.
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# Log to STDOUT (uses default formatting).
sh = logging.StreamHandler(stream=sys.stdout)
sh.setLevel(logging.INFO)
logger.addHandler(sh)

# Set Tensorflow logging level.
tf.get_logger().setLevel('ERROR') # 'INFO'

# List all GPUs visible to TensorFlow.
gpus = tf.config.list_physical_devices('GPU')
logger.info(f"Num GPUs Available: {len(gpus)}")
for gpu in gpus:
    logger.info(f"Name: {gpu.name}, Type: {gpu.device_type}")


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model',
        type=str,
        choices=[tup[0] for tup in inspect.getmembers(ml.models, inspect.ismodule)],
        help='Model name to train.'
    )
    parser.add_argument('--lr',
        type=float,
        required=True,
        help='Learning rate',
    )
    parser.add_argument('--epochs', 
        type=int,
        required=True,
        help='Number of training epochs.',
    )
    parser.add_argument('--in_seq_len', 
        type=int,
        required=True,
        help='Input sequence length.',
    )
    parser.add_argument('--out_seq_len', 
        type=int,
        required=True,
        help='Output sequence length.',
    )
    parser.add_argument('--shift', 
        type=int,
        required=True,
        help='Number of records to skip between data sequences.',
    )
    parser.add_argument('--split',
        type=float,
        nargs=3,
        default=(0.7,0.2,0.1),
        help='Data train/val/test split.',
    )
    parser.add_argument('--in_feat',
        type=str,
        nargs='+',
        default=['TEMP','PRES','Iws','Is','Ir'],
        help='Input feature list.',
    )
    parser.add_argument('--out_feat',
        type=str,
        nargs='+',
        default=['DEWP'],
        help='Output feature list.',
    )
    parser.add_argument('--batch_size',
        type=int,
        default=128,
        help='Batch size.',
    )
    parser.add_argument('--shuffle',
        action='store_true',
        help='Shuffle the dataset.',
    )
    opts = parser.parse_args()
    return opts


def main(opts: argparse.Namespace):
    # Set training strategy.
    strategy = tf.distribute.get_strategy()

    # Define model compilation parameters.
    # (i.e., loss, metrics, etc.)
    compile_params = dict(
        loss='mse',
        metrics=['mae', 'mape'],
    )

    def build_model_func() -> keras.Model:
        # Get build function for specific model.
        build_model = getattr(ml.models, opts.model).build_model
        model = build_model(
            in_seq_len=opts.in_seq_len,
            in_feat=len(opts.in_feat),
            out_feat=len(opts.out_feat),

            fc_units=[64,64],
            embed_dim=64,
            n_heads=8,
            ff_dim=256,
            dropout=0.1,
            n_encoders=3,
        )

        # Configure optimizer.
        # optim = keras.optimizers.Adam(learning_rate=lr_schedule)
        optim = keras.optimizers.Adam(learning_rate=opts.lr)

        # Compile the model.
        model.compile(
            optimizer=optim,
            loss=compile_params['loss'],
            metrics=compile_params['metrics'],
        )
        return model

    def dataset_loader_func(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return ml.datasets.beijingpm25.load_beijingpm25_ds(
            in_seq_len=opts.in_seq_len,
            out_seq_len=opts.out_seq_len,
            shift=opts.shift,
            in_feat=opts.in_feat,
            out_feat=opts.out_feat,
            split=opts.split,
            shuffle=opts.shuffle,
            path=config['roots']['dataset_root']/'beijing_pm25',
            batch_size=batch_size,
        )

    # Train and evaluate the model.
    model, hist, met = ml.training.train_evaluate_for_dataset(
        model_name=opts.model,
        build_model_func=build_model_func,
        dataset_loader_func=dataset_loader_func,
        metric_list=compile_params['metrics'],
        batch_size=opts.batch_size,
        strategy=strategy,
        epochs=opts.epochs,
        checkpoint_root=config['roots']['checkpoint_root'],
        callbacks=[TqdmCallback(verbose=2)]
    )
    model.summary(print_fn=logger.info)

    # Plot train/val performance.
    for key in compile_params['metrics']+['loss']:
        fig = ml.visualization.plot_metric(hist, key)
        fig.savefig(config['roots']['image_root']/f"{opts.model}_metric_{key}.png", bbox_inches='tight')
        # fig.show()

    # Load the data in dataframe form.
    df_train, df_val, df_test = ml.datasets.beijingpm25.load_beijingpm25_df(
        split=opts.split,
        path=config['roots']['dataset_root']/'beijing_pm25',
    )
    # Load the data in dataset form.
    dataset_train, dataset_val, dataset_test = dataset_loader_func(opts.batch_size)
    # Data keys.
    keys = [
        'year',
        'month',
        'day',
        'hour',
        'pm2.5',
        'DEWP',
        'TEMP',
        'PRES',
        'Iws',
        'Is',
        'Ir',
    ]
    # Normalize specific keys.
    train_mean = df_train[keys].mean()
    train_std = df_train[keys].std()
    df_train[keys] = (df_train[keys] - train_mean)/train_std
    df_val[keys] = (df_val[keys] - train_mean)/train_std
    df_test[keys] = (df_test[keys] - train_mean)/train_std
    # Evaluate the model on the train/val/test data.
    train_pred = model.predict(dataset_train)
    val_pred = model.predict(dataset_val)
    test_pred = model.predict(dataset_test)
    # Plot the model predictions of the dataset.
    # Create figure for each data set.
    figs = {}
    labels = ['train', 'val', 'test']
    for l, label in enumerate(labels):
        fig = ml.visualization.plot_input_output(
            df=locals()[f"df_{label}"],
            pred=locals()[f"{label}_pred"],
            in_seq_len=opts.in_seq_len,
            out_seq_len=opts.out_seq_len,
            shift=opts.shift,
            in_feat=opts.in_feat,
            out_feat=opts.out_feat,
            x_key='datetime',
        )
        # fig.suptitle(f"{label[0].upper()}{label[1:]} Data", fontsize='xx-large')
        figs[label] = fig
    for name, fig in figs.items():
        fig.savefig(config['roots']['image_root']/f"{opts.model}_io_{name}.png", bbox_inches='tight')
        # fig.show()


if __name__ == '__main__':
    opts = get_opts()
    main(opts)

# # Setup paths.
# DATASET_ROOT = Path('~/research/makassar/datasets').expanduser()
# if not DATASET_ROOT.exists(): raise ValueError(f"Dataset root directory does not exist at {DATASET_ROOT}")
# PROJECT_ROOT = Path('~/research/makassar').expanduser()
# CHECKPOINT_ROOT = PROJECT_ROOT / 'checkpoints'
# IMAGE_ROOT = PROJECT_ROOT / 'images'
# TABLE_ROOT = PROJECT_ROOT / 'tables'
# HP_TUNING_ROOT = PROJECT_ROOT / 'hp_tuning'
# KERAS_TUNER_PATH = PROJECT_ROOT / 'keras_tuner'

# # Ensure some directories exist.
# PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
# CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
# IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
# TABLE_ROOT.mkdir(parents=True, exist_ok=True)
# HP_TUNING_ROOT.mkdir(parents=True, exist_ok=True)
# KERAS_TUNER_PATH.mkdir(parents=True, exist_ok=True)