from __future__ import annotations
import ast
import json
import keras_tuner
import logging
from typing import Callable
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
import tensorflow.keras as keras

from .training import ensure_path, load_metrics, load_history, load_trained_model, train_evaluate_model

# Create logger for module.
logger = logging.getLogger(__name__)

def str_eval_wrapper(func):
    """Helper that converts function arguments from strings using literal evaluation."""
    def wrap(*args, **kwargs):
        # Handle certain variables being a string (supports Keras hypertuning).
        for i in range(len(args)):
            if isinstance(args[i], str):
                args[i] = ast.literal_eval(args[i])
        for k in kwargs:
            if isinstance(kwargs[k], str):
                kwargs[k] = ast.literal_eval(kwargs[k])
        # Evaluate function with converted arguments.
        return func(*args, **kwargs)
    return wrap


def _config2dict(config: dict, parse_node: Callable[[object], object]) -> dict:

    # Dictionary of parameters.
    params = dict()

    # Parse model.
    if 'parameters' in config['model']:
        for key,val in config['model']['parameters'].items():
            params[key] = parse_node(key, val)

    # Parse training optimier.
    if 'optimizer' in config['train']:
        if 'parameters' in config['train']['optimizer']:
            params['optimizer'] = parse_node('optimizer', config['train']['optimizer']['name'])
            for key,val in config['train']['optimizer']['parameters'].items():
                params[key] = parse_node(key, val)

    return params


def config2parameterdict(config: dict) -> dict:
    """Converts configuration dictionary into parameters acceptable for `ParameterGrid`."""

    def parse_node(key: str, node: object) -> object:
        """Helper to parse a configuration node."""
        # Configuration.
        if isinstance(node, dict):
            # Single value, return as list of 1 element.
            if 'value' in node:
                return [node['value']]
            # Multiple values, return as-is.
            elif 'values' in node:
                return node['values']
            # Range of values.
            elif 'range' in node:
                return list(np.arange(node['range']['min'], node['range']['max'], node['range']['step']))
            # Process dictionary as separate keys as separate unique entries.
            else:
                return [{k:v} for k,v in node.items()]
        # Single value.
        else:
            return [node]

    return _config2dict(config, parse_node)


def config2hyperparameterdict(config: dict) -> dict:
    """Converts configuration dictionary into parameters acceptable for Keras Tuner."""

    # Create hyperparameter object.
    hp = keras_tuner.HyperParameters()

    def parse_node(key: str, node: object) -> object:
        """Helper to parse a configuration node."""
        # Configuration.
        if isinstance(node, dict):
            # Single value, return as fixed element.
            if 'value' in node:
                return hp.Fixed(
                    name=key,
                    value=node['value'],
                )
            # Multiple values, return choice of list of strings.
            elif 'values' in node:
                return hp.Choice(
                    name=key,
                    values=[str(item) for item in node['values']],
                )
            # Range of values.
            elif 'range' in node:
                return hp.Float(
                    name=key,
                    min_value=node['range']['min'],
                    max_value=node['range']['max'],
                    step=node['range']['step'],
                )
        # Single value.
        else:
            return hp.Fixed(
                    name=key,
                    value=node,
                )

    return _config2dict(config, parse_node)



def hp_gridsearch(
    model_name: str,
    params: dict,
    build_model_func: Callable[[dict], keras.Model], # this function must compile the model too.
    dataset_loader_func: Callable[[int], tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]],
    batch_size: int = 128,
    strategy: tf.distribute.Strategy = tf.distribute.get_strategy(),
    epochs: int = 10,
    tuning_root: str = None,
    callbacks: list = [],
    ) -> tuple[keras.models.Model, pd.DataFrame, dict, pd.DataFrame, list[pd.DataFrame]]:
    """Train and evaluate a model on a given dataset.

    If checkpoint exists then the model is loaded in place of training.
    """
    # Create the tuning directory for this model if it does not already exist.
    tuning_model_root = ensure_path(tuning_root)/model_name
    tuning_model_root.mkdir(parents=True, exist_ok=True)

    # Maximize batch size efficiency using distributed strategy.
    batch_size_per_replica = batch_size
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # Build the parameter grid.
    grid = ParameterGrid(params)
    n_grid = len(grid)
    logger.info(f"[{model_name}] Evaluating {n_grid} hyperparameter combinations")

    # Validate any previous parameter grid runs to ensure parameters were the same.
    # If same, then do nothing. If different, then remove all old contents.
    parameter_grid_path = tuning_model_root/'parameter_grid.json'
    if parameter_grid_path.exists():
        with open(tuning_model_root/'parameter_grid.json', 'r') as f:
            old_grid = json.load(f)

        # New grid is different, so overwrite everything.
        new_grid = list(grid)
        if old_grid != new_grid:

            # Recreate tuning directory.
            shutil.rmtree(tuning_model_root)
            tuning_model_root.mkdir(parents=True, exist_ok=True)

    # Update parameter grid file.
    with open(tuning_model_root/'parameter_grid.json', 'w') as f:
        json.dump(list(grid), f, default=lambda o: '<not serializable>')

    # Pre-load dataset for faster training.
    with strategy.scope():
        dataset_train, dataset_val, dataset_test = dataset_loader_func(batch_size=batch_size)

    # Iterate over the parameter grid to train the models.
    df_results: list[dict] = []
    histories: list = []
    for i, p in enumerate(grid):

        # Build current model name string.
        cur_model_name = f"model_{i}"
        logger.info(f"[{model_name}, {cur_model_name}] Parameters: {p}")

        # Build paths for train/eval checkpoint, history, and metrics.
        checkpoint_path = tuning_model_root/f"{cur_model_name}.h5"
        history_path = tuning_model_root/f"{cur_model_name}_history.csv"
        metrics_path = tuning_model_root/f"{cur_model_name}_metrics.json"
        hparams_path = tuning_model_root/f"{cur_model_name}_hparams.json"

        # Validate any existing hyperparameter combinations.
        do_load = False
        if checkpoint_path.exists() and history_path.exists() and metrics_path.exists() and hparams_path.exists():
            with open(hparams_path, 'r') as f:
                hp = json.load(f)
            do_load = (hp == p)

        # Load model from checkpoint.
        if do_load:
            logger.info(f"[{model_name}, {cur_model_name}] Loading from save data")
            hist = load_history(history_path)
            met = load_metrics(metrics_path)

        # Train the model if no checkpoint exists.
        else:
            logger.info(f"[{model_name}, {cur_model_name}] Training new model: {epochs=}, {batch_size=}")

            # Save hyperparameters to file.
            with open(hparams_path, 'w') as f:
                json.dump(p, f, default=lambda o: '<not serializable>')

            # Build the model.
            with strategy.scope():
                logger.info(f"[{model_name}, {cur_model_name}] Model building...")

                # # Load the dataset.
                # dataset_train, dataset_val, dataset_test = dataset_loader_func(batch_size=batch_size)

                # Build and compile model.
                model = build_model_func(p)

                logger.info(f"[{model_name}, {cur_model_name}] Model built: {p}")

            # Train and evaluate the model and get the trained model, history, and metrics.
            _, hist, met = train_evaluate_model(
                model,
                datagen_train=dataset_train,
                datagen_val=dataset_val,
                datagen_test=dataset_test,
                epochs=epochs,
                checkpoint_path=checkpoint_path,
                history_path=history_path,
                metrics_path=metrics_path,
                callbacks=callbacks,
            )

        logger.info(f"[{model_name}, {cur_model_name}] metrics: {met}")

        # Append to lists.
        histories.append(hist)

        # Populate results list with the current parameters and metrics.
        df_results.append({
            'model': i,
            **met,
            **p,
        })

    # Build dataframe using results.
    df = pd.DataFrame(df_results)

    # Now only load the best model.
    best_idx = df[['val_loss']].idxmin().values[0]
    best_model_name = f"model_{best_idx}"
    checkpoint_path = tuning_model_root/f"{best_model_name}.h5"
    history_path = tuning_model_root/f"{best_model_name}_history.csv"
    metrics_path = tuning_model_root/f"{best_model_name}_metrics.json"
    params = grid[best_idx]
    logger.info(f"[{model_name}] Loading best model {best_idx}: {params}")
    model = load_trained_model(checkpoint_path)
    hist = load_history(history_path)
    met = load_metrics(metrics_path)

    return model, hist, met, params, df, histories