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

# Complete imports.
import argparse
import logging
import makassar_ml as ml
import shutil
import tensorflow as tf
import tensorflow.keras as keras
import yaml

# Set random seeds.
SEED = 0
tf.random.set_seed(SEED) # Only this works on ARC (since tensorflow==2.4).

# Setup logging (useful for ARC systems).
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
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
    parser.add_argument('config',
        type=str,
        help='Configuration file.'
    )
    parser.add_argument('--force',
        action='store_true',
        help='Force train a new model by removing any existing checkpoints.'
    )
    opts = parser.parse_args()
    return opts


def main(config: dict, force: bool = False):

    # Force re-training by removing old checkpoint if necessary.
    if force:
        logger.info('Force training')
        path = Path(config['roots']['checkpoint_root'])/config['model']['name']
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint directory: {path}")

    # Set training strategy.
    strategy = tf.distribute.get_strategy()

    def build_model_func() -> keras.Model:
        # Get build function for specific model.
        build_model = getattr(ml.models, config['model']['name']).build_model
        model = build_model(
            **config['model']['params']
        )

        # Configure optimizer.
        # optim = keras.optimizers.Adam(learning_rate=lr_schedule)
        optim = keras.optimizers.get({
        'class_name': config['train']['optimizer']['name'],
            'config': config['train']['optimizer']['params'],
        })

        # Compile the model.
        model.compile(
            optimizer=optim,
            **config['train']['compile'],
        )
        return model

    def dataset_loader_func(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return ml.datasets.beijingpm25.load_beijingpm25_ds(
            **config['dataset']['params'],
            batch_size=batch_size,
        )

    # Train and evaluate the model.
    model, hist, met = ml.training.train_evaluate_for_dataset(
        model_name=config['model']['name'],
        build_model_func=build_model_func,
        dataset_loader_func=dataset_loader_func,
        metric_list=config['train']['compile']['metrics'],
        batch_size=config['train']['batch_size'],
        strategy=strategy,
        epochs=config['train']['epochs'],
        checkpoint_root=config['roots']['checkpoint_root'],
    )
    model.summary(print_fn=logger.info)

    # Plot train/val performance.
    for key in config['train']['compile']['metrics']+['loss']:
        fig = ml.visualization.plot_metric(hist, key)
        path = Path(config['roots']['image_root'])/f"{config['model']['name']}_metric_{key}.png"
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"{path}")
        # fig.show()

    # Load the data in dataframe form.
    df_train, df_val, df_test = ml.datasets.beijingpm25.load_beijingpm25_df(
        split=config['dataset']['params']['split'],
        path=config['dataset']['params']['path'],
    )
    # Load the data in dataset form.
    dataset_train, dataset_val, dataset_test = dataset_loader_func(config['train']['batch_size'])
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
            in_seq_len=config['dataset']['params']['in_seq_len'],
            out_seq_len=config['dataset']['params']['out_seq_len'],
            shift=config['dataset']['params']['shift'],
            in_feat=config['dataset']['params']['in_feat'],
            out_feat=config['dataset']['params']['out_feat'],
            x_key='datetime',
        )
        # fig.suptitle(f"{label[0].upper()}{label[1:]} Data", fontsize='xx-large')
        figs[label] = fig
    for name, fig in figs.items():
        path = Path(config['roots']['image_root'])/f"{config['model']['name']}_io_{name}.png"
        fig.savefig(path, bbox_inches='tight')
        logger.info(f"{path}")
        # fig.show()


if __name__ == '__main__':
    # Get program options.
    opts = get_opts()

    # Load YAML config.
    config_path = Path(opts.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.load(f, Loader=ml.yaml.EnvVarLoader)

    # import json
    # print(json.dumps(config, indent=4))

    # Run main training function.
    main(config, opts.force)