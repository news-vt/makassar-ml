from __future__ import annotations
from functools import partial

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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shutil
import tensorflow as tf
import tensorflow.keras as keras
import yaml
sns.set()

# Set random seeds.
SEED = 0
tf.random.set_seed(SEED) # Only this works on ARC (since tensorflow==2.4).

# Setup logging (useful for ARC systems).
# logger = logging.getLogger(__name__)
logger = logging.getLogger()
logger.propagate = False
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
    parser.add_argument('--count',
        action='store_true',
        help='Count number of combinations.'
    )
    parser.add_argument('--dump-grid',
        action='store_true',
        help='Dump parameter grid as JSON.'
    )
    parser.add_argument('--dump-json',
        action='store_true',
        help='Dump configuration as JSON.'
    )
    parser.add_argument('--no-plot',
        action='store_true',
        help='Do not plot after tuning.'
    )
    opts = parser.parse_args()
    return opts


def main(
    config: dict,
    force: bool = False,
    no_plot: bool = False,
    ):

    # Validate model name.
    if not hasattr(ml.models, config['model']['name']):
        raise ValueError(f"unknown model {config['model']['name']}")

    # Validate dataset name.
    if not hasattr(ml.datasets, config['dataset']['name']):
        raise ValueError(f"unknown dataset {config['dataset']['name']}")

    # Force re-training by removing old checkpoint if necessary.
    if force:
        path = Path(config['roots']['hp_tuning_root'])/config['model']['name']
        if path.exists():
            logger.info('Force training')
            shutil.rmtree(path)
            logger.info(f"Removed old tuning directory: {path}")

    # Set training strategy.
    strategy = tf.distribute.get_strategy()

    def build_model_func(params: dict) -> keras.Model:
        # Create copy of parameter dictionary.
        model_params = dict(**params)

        # Configure optimizer.
        optimizer_config = dict()
        if 'lr' in model_params:
            if isinstance(model_params['lr'], (int, float)):
                optimizer_config['lr'] = model_params.pop('lr')
            elif isinstance(model_params['lr'], dict):
                decayfunc_name = list(model_params['lr'].keys())[0]
                decayfunc_params = model_params['lr'][decayfunc_name]
                if hasattr(keras.experimental, decayfunc_name):
                    optimizer_config['lr'] = getattr(keras.experimental, decayfunc_name)(**decayfunc_params)
                elif hasattr(keras.optimizers.schedules, decayfunc_name):
                    optimizer_config['lr'] = getattr(keras.optimizers.schedules, decayfunc_name)(**decayfunc_params)
                del model_params['lr'] # Remove.
        optimizer_class_name = model_params.pop('optimizer')
        optim = keras.optimizers.get({
        'class_name': optimizer_class_name,
            'config': optimizer_config,
        })

        # Get build function for specific model.
        build_model = getattr(
            ml.models,
            config['model']['name'],
        )
        model = build_model(
            **model_params,
        )

        # Compile the model.
        model.compile(
            optimizer=optim,
            **config['train']['compile'],
        )
        return model

    def dataset_loader_func(batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

        # Load dataset from package.
        if hasattr(ml.datasets, config['dataset']['name']):
            func = getattr(ml.datasets, config['dataset']['name']).load_data
            return func(
                **config['dataset']['parameters'],
                batch_size=batch_size,
            )

    # Create callback list if any were specified.
    callbacks = []
    if 'callbacks' in config['train']:
        for key, cb_params in config['train']['callbacks'].items():
            if hasattr(ml.callbacks, key):
                callbacks.append(getattr(ml.callbacks, key)(**cb_params))
            elif hasattr(keras.callbacks, key):
                if key == 'LearningRateScheduler':
                    callbacks.append(
                        keras.callbacks.LearningRateScheduler(
                            partial(
                                getattr(ml.schedules, cb_params['schedule']),
                                **cb_params.get('parameters', {})
                            )
                        )
                    )
                else:
                    callbacks.append(
                        getattr(keras.callbacks, key)(**cb_params)
                    )

    # Convert config into parameter dictionary.
    parameterdict = ml.tuning.config2parameterdict(config)

    # Train and evaluate the model.
    model, hist, met, params, df, allhist = ml.tuning.hp_gridsearch(
        model_name=config['model']['name'],
        params=parameterdict,
        build_model_func=build_model_func,
        dataset_loader_func=dataset_loader_func,
        metric_list=config['train']['compile']['metrics'],
        batch_size=config['train']['batch_size'],
        strategy=strategy,
        epochs=config['train']['epochs'],
        tuning_root=config['roots']['hp_tuning_root'],
        callbacks=callbacks,
    )
    model.summary(print_fn=logger.info)

    logger.info(f"Tuning Results:")
    # Build the resulting table header.
    table_header = ['model']
    for m in ['loss']+config['train']['compile']['metrics']:
        table_header.append(f"{m}")
        table_header.append(f"val_{m}")
        table_header.append(f"test_{m}")
    table_header.extend(list(parameterdict))
    # Log results as CSV to console.
    csv_df = df[table_header].sort_values(by='val_loss', ascending=True)
    logger.info(csv_df.to_string(index=False))
    # Log results as CSV to file.
    csv_path = Path(config['roots']['hp_tuning_root'])/config['model']['name']/f"tuning_results.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info(csv_path)

    # Export the dataframe to LaTeX using custom style.
    latex_df = csv_df
    latex_df.columns = latex_df.columns.map(lambda x: x.replace('_', '\_')) # Escape the header names too.
    styler = latex_df.style
    styler = styler.format(str, escape='latex') # Default is to convert all cells to their string representation.
    subset = []
    for m in ['loss']+config['train']['compile']['metrics']:
        if m in set(latex_df.columns):
            subset.append(f"{m}")
            subset.append(f"val\_{m}")
            subset.append(f"test\_{m}")
    styler = styler.format(formatter='{:.4f}', subset=subset)
    styler = styler.highlight_min(subset=subset, axis=0, props='textbf:--rwrap;')
    styler = styler.hide(axis=0) # Hide the index.
    latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_results.tex"
    styler.to_latex(
        buf=latex_path,
        hrules=True,
    )
    latex_df.columns = latex_df.columns.map(lambda x: x.replace('\_', '_')) # Convert header names back.
    logger.info(latex_path)

    ###
    # Plotting.
    ###

    if not no_plot:

        # Plot train/val performance for best model.
        for key in config['train']['compile']['metrics']+['loss']:
            fig = ml.visualization.plot_metric(hist, key)
            path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_metric_{key}_best.png"
            fig.savefig(path, bbox_inches='tight')
            logger.info(path)
            # fig.show()

        # Plot train/val/test metrics for all models.
        n_hist = len(allhist)
        color = plt.cm.rainbow(np.linspace(0, 1, n_hist))
        for key in config['train']['compile']['metrics']+['loss']:
            fig = plt.figure(figsize=(8,6))
            for i, (h, c) in enumerate(zip(allhist, color)):
                plt.plot(h[key], label=f"model {i} train", color=c, linestyle='-')
                plt.xlim(0, len(h[key])-1)
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.legend(loc='center left', ncol=2, bbox_to_anchor=(1.04,0.5))
            path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_metric_{key}_all_train.png"
            fig.savefig(path, bbox_inches='tight')
            logger.info(path)
            # fig.show()

            fig = plt.figure(figsize=(8,6))
            for i, (h, c) in enumerate(zip(allhist, color)):
                plt.plot(h[f'val_{key}'], label=f"model {i} val", color=c, linestyle='-')
                plt.xlim(0, len(h[key])-1)
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.legend(loc='center left', ncol=2, bbox_to_anchor=(1.04,0.5))
            path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_metric_{key}_all_val.png"
            fig.savefig(path, bbox_inches='tight')
            logger.info(path)
            # fig.show()

        # # Load the data in dataframe form.
        # df_train, df_val, df_test = ml.datasets.beijingpm25.load_beijingpm25_df(
        #     split=config['dataset']['parameters']['split'],
        #     path=config['dataset']['parameters']['path'],
        # )
        # # Load the data in dataset form.
        # dataset_train, dataset_val, dataset_test = dataset_loader_func(config['train']['batch_size'])
        # # Data keys.
        # keys = [
        #     'year',
        #     'month',
        #     'day',
        #     'hour',
        #     'pm2.5',
        #     'DEWP',
        #     'TEMP',
        #     'PRES',
        #     'Iws',
        #     'Is',
        #     'Ir',
        # ]
        # # Normalize specific keys.
        # train_mean = df_train[keys].mean()
        # train_std = df_train[keys].std()
        # df_train[keys] = (df_train[keys] - train_mean)/train_std
        # df_val[keys] = (df_val[keys] - train_mean)/train_std
        # df_test[keys] = (df_test[keys] - train_mean)/train_std
        # # Evaluate the model on the train/val/test data.
        # train_pred = model.predict(dataset_train)
        # val_pred = model.predict(dataset_val)
        # test_pred = model.predict(dataset_test)
        # # Plot the model predictions of the dataset.
        # # Create figure for each data set.
        # figs = {}
        # labels = ['train', 'val', 'test']
        # for l, label in enumerate(labels):
        #     fig = ml.visualization.plot_input_output(
        #         df=locals()[f"df_{label}"],
        #         pred=locals()[f"{label}_pred"],
        #         in_seq_len=config['dataset']['parameters']['in_seq_len'],
        #         out_seq_len=config['dataset']['parameters']['out_seq_len'],
        #         shift=config['dataset']['parameters']['shift'],
        #         in_feat=config['dataset']['parameters']['in_feat'],
        #         out_feat=config['dataset']['parameters']['out_feat'],
        #         x_key='datetime',
        #     )
        #     # fig.suptitle(f"{label[0].upper()}{label[1:]} Data", fontsize='xx-large')
        #     figs[label] = fig
        # for name, fig in figs.items():
        #     path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_io_{name}.png"
        #     fig.savefig(path, bbox_inches='tight')
        #     logger.info(f"{path}")
        #     # fig.show()


if __name__ == '__main__':
    # Get program options.
    opts = get_opts()

    # Load YAML config.
    config_path = Path(opts.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.load(f, Loader=ml.yaml.ConfigLoader)

    # Count number of parameter combinations.
    if opts.count:
        from sklearn.model_selection import ParameterGrid
        grid = ParameterGrid(ml.tuning.config2parameterdict(config))
        logger.info(len(grid))

    # Dump parameter grid as JSON.
    elif opts.dump_grid:
        import json
        from sklearn.model_selection import ParameterGrid
        grid = ParameterGrid(ml.tuning.config2parameterdict(config))
        logger.info(json.dumps(list(grid), indent=4))

    # Dump configuration as JSON.
    elif opts.dump_json:
        import json
        logger.info(json.dumps(config, indent=4))

    # Run main tuning function.
    else:
        main(
            config=config,
            force=opts.force,
            no_plot=opts.no_plot,
        )