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
from cycler import cycler
from functools import reduce
from itertools import cycle
import logging
import makassar_ml as ml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def build_latex_tuning_results(
    latex_config: dict,
    df: pd.DataFrame,
    metric_keys: list,
    latex_path: Path,
    ):
    
    if 'header' in latex_config:
        table_header = latex_config.pop('header')
    else:
        table_header = ['model'] + metric_keys
    latex_df = df[table_header]
    if 'sort_values' in latex_config:
        latex_df = latex_df.sort_values(**latex_config.pop('sort_values', {}))
    latex_df.columns = latex_df.columns.map(lambda x: x.replace('_', '\_')) # Escape the header names too.
    styler = latex_df.style
    styler = styler.format(str, escape='latex') # Default is to convert all cells to their string representation.
    styler = styler.format(
        formatter='{:.4f}',
        subset=[key.replace('_', '\_') for key in metric_keys if key in table_header],
    )
    subset = [key.replace('_', '\_') for key in metric_keys if key in table_header and 'accuracy' in key]
    if len(subset) > 0:
        styler = styler.highlight_max(
            subset=subset,
            axis=0,
            props='textbf:--rwrap;',
        )
    subset = [key.replace('_', '\_') for key in metric_keys if key in table_header and 'accuracy' not in key]
    if len(subset) > 0:
        styler = styler.highlight_min(
            subset=subset, 
            axis=0,
            props='textbf:--rwrap;',
        )
    styler = styler.hide(axis=0) # Hide the index.
    # latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_results.tex"
    latex_string = styler.to_latex(
        # buf=latex_path,
        hrules=True,
        label=f"tab:{latex_path.stem}",
        **latex_config,
    )
    if 'longtable' in latex_string:
        latex_string = f"""
\\begingroup
\\renewcommand\\arraystretch{{0.5}}
{latex_string}
\endgroup
"""
    with open(latex_path, 'w') as f:
        f.write(latex_string)
    latex_df.columns = latex_df.columns.map(lambda x: x.replace('\_', '_')) # Convert header names back.
    logger.info(latex_path)


def build_latex_tuning_parameters(
    latex_config: dict,
    df: pd.DataFrame,
    metric_keys: list,
    latex_path: Path,
    ):

    if 'header' in latex_config:
        table_header = latex_config.pop('header')
    else:
        table_header = ['model']
        table_header.extend(sorted(set(df.keys()) - set(metric_keys) - set(table_header)))
    latex_df = df[table_header]
    if 'sort_values' in latex_config:
        latex_df = latex_df.sort_values(**latex_config.pop('sort_values', {}))
    else:
        latex_df = latex_df.sort_values(by='model', ascending=True)

    latex_df.columns = latex_df.columns.map(lambda x: x.replace('_', '\_')) # Escape the header names too.
    styler = latex_df.style
    styler = styler.format(str, escape='latex') # Default is to convert all cells to their string representation.
    def latex_lr_formatter(val) -> str:
        # if isinstance(val, (int, float)):
        #     # return f"{val:.4f}"
        #     return f"{val:.4f}"
        if isinstance(val, dict):
            name = list(val.keys())[0]
            return name
            # s = ', '.join([f"{k}={v}" for k,v in val[name].items()])
            # return f"${name}({s})$".replace('_', '\_')
        else:
            return str(val)
    styler = styler.format(
        formatter=latex_lr_formatter,
        subset=[key for key in table_header if key == 'lr' or key == 'learning_rate'],
    )
    styler = styler.hide(axis=0) # Hide the index.
    # latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_parameters.tex"
    latex_string = styler.to_latex(
        # buf=latex_path,
        hrules=True,
        label=f"tab:{latex_path.stem}",
        **latex_config,
    )
    if 'longtable' in latex_string:
        latex_string = f"""
\\begingroup
\\renewcommand\\arraystretch{{0.5}}
{latex_string}
\endgroup
"""
    with open(latex_path, 'w') as f:
        f.write(latex_string)
    latex_df.columns = latex_df.columns.map(lambda x: x.replace('\_', '_')) # Convert header names back.
    logger.info(latex_path)


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
        batch_size=config['train']['batch_size'],
        strategy=strategy,
        epochs=config['train']['epochs'],
        tuning_root=config['roots']['hp_tuning_root'],
        callbacks=callbacks,
    )
    model.summary(print_fn=logger.info)



    # Create list of keys.
    metric_keys_raw = list(set(hist.keys()) - set(['epoch']))
    metric_keys_base = [key for key in metric_keys_raw if not key.startswith('val_')] # Isolate base metrics.
    # Generate list of keys in a specific order.
    metric_keys_sorted = []
    for metric_base_name in ['loss', 'accuracy']:
        # Add the exact name first if it appears.
        if metric_base_name in metric_keys_base:
            metric_keys_sorted.append(metric_base_name)
        # Add variations of the name.
        for key in metric_keys_base:
            if metric_base_name in key and metric_base_name != key:
                metric_keys_sorted.append(key)
    metric_keys_sorted.extend(sorted(set(metric_keys_base) - set(metric_keys_sorted)))

    # Add val/test prefixes.
    metric_keys = []
    for key in metric_keys_sorted:
        metric_keys.append(key)
        metric_keys.append(f"val_{key}")
        metric_keys.append(f"test_{key}")

    logger.info(f"Tuning Results:")

    # Log results as CSV to console and a file.
    table_header = ['model'] + metric_keys
    table_header.extend(sorted(set(df.keys()) - set(table_header)))
    csv_df = df[table_header].sort_values(by='val_loss', ascending=True)
    logger.info(csv_df.to_string(index=False))
    csv_path = Path(config['roots']['hp_tuning_root'])/config['model']['name']/f"tuning_results.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info(csv_path)

    # Export the performance results to LaTeX.
    latex_config = config.get('latex_table_results', {})
    if isinstance(latex_config, list):
        for i, cfg in enumerate(latex_config):
            latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_results_{i}.tex"
            build_latex_tuning_results(
                latex_config=cfg,
                df=df,
                metric_keys=metric_keys,
                latex_path=latex_path,
            )
    else:
        latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_results.tex"
        build_latex_tuning_results(
            latex_config=latex_config,
            df=df,
            metric_keys=metric_keys,
            latex_path=latex_path,
        )


    # Export the parameters to LaTeX.
    latex_config = config.get('latex_table_parameters', {})
    if isinstance(latex_config, list):
        for i, cfg in enumerate(latex_config):
            latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_parameters_{i}.tex"
            build_latex_tuning_parameters(
                latex_config=cfg,
                df=df,
                metric_keys=metric_keys,
                latex_path=latex_path,
            )
    else:
        latex_path = Path(config['roots']['table_root'])/f"{config['model']['name']}_tuning_parameters.tex"
        build_latex_tuning_parameters(
            latex_config=latex_config,
            df=df,
            metric_keys=metric_keys,
            latex_path=latex_path,
        )


    ###
    # Plotting.
    ###

    if not no_plot:

        # Get index of best model.
        best_idx = df[['val_loss']].idxmin().values[0]

        # Plot train/val performance for best model.
        fig, ax = plt.subplots(nrows=len(metric_keys_base), ncols=1, figsize=(10,len(metric_keys_base)*3), sharex=True, constrained_layout=True)
        for j, key in enumerate(sorted(metric_keys_base)):
            ax[j].plot(hist[key], label=f'model {best_idx} train', linestyle='-')
            ax[j].plot(hist[f"val_{key}"], label=f'model {best_idx} val', linestyle='--')
            ax[j].set_xlim(0, len(hist[key])-1)
            if j == len(metric_keys_base)-1:
                ax[j].set_xlabel('epoch')
            ax[j].set_ylabel(key)
            # ax[j].legend(loc='upper left')

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5,0.0))
        path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_metric_best.png"
        fig.savefig(path, bbox_inches='tight')
        logger.info(path)
            # fig.show()

        # Set line style cycling.
        n_hist = len(allhist)
        cycles = [
            cycler('linestyle', ['-', '--', ':', '-.']),
            cycler('marker', ['o', '+', 'x', '*', 'd', 'X', 's', '^']),
        ]
        sty_cycle = reduce(lambda x,y: x*y, cycles)
        color_cycle = cycler('color', plt.rcParams['axes.prop_cycle'].by_key()['color'])
        # color_cycle = cycler('color', plt.cm.rainbow(np.linspace(0, 1, n_hist)))
        styles = [{**sty,**c} for _, sty, c in zip(range(len(allhist)), cycle(sty_cycle), cycle(color_cycle))] # Create definitive list of styles for lookup.

        # Plot train/val/test metrics for all models.
        nrows = len(metric_keys_base)
        ncols = 2
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,len(metric_keys_base)*3), sharex='col', sharey='none', constrained_layout=True)
        for j, key in enumerate(sorted(metric_keys_base)):
            for i, (h, sty) in enumerate(zip(allhist, styles)):
                # Train.
                ax[j,0].plot(h[key], label=f"model {i}", markersize=5, **sty)
                ax[j,0].set_xlim(0, len(h[key])-1)
                if j == len(metric_keys_base)-1:
                    ax[j,0].set_xlabel('epoch')
                ax[j,0].set_ylabel(key)
                if j == 0:
                    ax[j,0].set_title('Training')

                # Val.
                ax[j,1].plot(h[f'val_{key}'], label=f"model {i}", markersize=5, **sty)
                ax[j,1].set_xlim(0, len(h[key])-1)
                if j == len(metric_keys_base)-1:
                    ax[j,1].set_xlabel('epoch')
                if j == 0:
                    ax[j,1].set_title('Validation')
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=9, bbox_to_anchor=(0.5,0.0))
        path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_metric_all.png"
        fig.savefig(path, bbox_inches='tight')
        logger.info(path)


        ###### top 4

        if 'highlight_plot' in config:

            df_highlight = df

            if 'models' in config['highlight_plot']:
                df_highlight = df_highlight[df_highlight['model'].isin(config['highlight_plot']['models'])]

            elif 'nsmallest' in config['highlight_plot']:
                df_highlight = df_highlight.nsmallest(**config['highlight_plot']['nsmallest'])

            elif 'nlargest' in config['highlight_plot']:
                df_highlight = df_highlight.nlargest(**config['highlight_plot']['nlargest'])

            # Plot train/val/test metrics for all models.
            nrows = len(metric_keys_base)
            ncols = 2
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,len(metric_keys_base)*3), sharex='col', sharey='none', constrained_layout=True)
            for j, key in enumerate(sorted(metric_keys_base)):
                for _, row in df_highlight.iterrows():

                    # Gather model index, training history, and style.
                    i = row['model']
                    sty = styles[i]
                    h = allhist[i]

                    # Train.
                    ax[j,0].plot(h[key], label=f"model {i}", markersize=5, **sty)
                    ax[j,0].set_xlim(0, len(h[key])-1)
                    if j == len(metric_keys_base)-1:
                        ax[j,0].set_xlabel('epoch')
                    ax[j,0].set_ylabel(key)
                    if j == 0:
                        ax[j,0].set_title('Training')

                    # Val.
                    ax[j,1].plot(h[f'val_{key}'], label=f"model {i}", markersize=5, **sty)
                    ax[j,1].set_xlim(0, len(h[key])-1)
                    if j == len(metric_keys_base)-1:
                        ax[j,1].set_xlabel('epoch')
                    if j == 0:
                        ax[j,1].set_title('Validation')
            handles, labels = ax[0,0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=9, bbox_to_anchor=(0.5,0.0))
            path = Path(config['roots']['image_root'])/f"tuned_{config['model']['name']}_metric_highlight.png"
            fig.savefig(path, bbox_inches='tight')
            logger.info(path)


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