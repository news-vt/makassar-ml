from __future__ import annotations
import ast
import json
import logging
from typing import Callable
import pandas as pd
from pathlib import Path
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


def hp_gridsearch(
    model_name: str,
    build_model_func: Callable[[dict], keras.Model], # this function must compile the model too.
    dataset_loader_func: Callable[[int], tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]],
    metric_list: list[str],
    batch_size: int = 128,
    strategy: tf.distribute.Strategy = tf.distribute.get_strategy(),
    epochs: int = 10,
    tuning_root: str = None,
    table_root: str = None,
    callbacks: list = [],

    # params: dict,
    # compile_params: dict,
    # in_seq_len: int,
    # out_seq_len: int,
    # shift: int,
    # split: tuple[float, float, float],
    # in_feat: list[str],
    # out_feat: list[str],
    # tuning_path: str,
    # batch_size: int = 128,
    # shuffle: bool = False,
    # strategy: tf.distribute.Strategy = tf.distribute.get_strategy(),
    # epochs: int = 10,
    table_header: list = None,
    sort_cols: str|list[str] = None,
    sort_ascending: bool = True,
    # table_omit_cols: str|list[str] = None,
    ) -> tuple[keras.models.Model, dict, dict, pd.DataFrame]:
    """Train and evaluate a model on a given dataset.

    If checkpoint exists then the model is loaded in place of training.
    """
    # Create the tuning directory for this model if it does not already exist.
    tuning_model_root = ensure_path(tuning_root)/model_name
    tuning_model_root.mkdir(parents=True, exist_ok=True)



    # Maximize batch size efficiency using distributed strategy.
    batch_size_per_replica = batch_size
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # # Load the dataset.
    # with strategy.scope():
    #     dataset_train, dataset_val, dataset_test = load_beijingpm25_ds(
    #         in_seq_len=in_seq_len,
    #         out_seq_len=out_seq_len,
    #         shift=shift,
    #         in_feat=in_feat,
    #         out_feat=out_feat,
    #         split=split,
    #         shuffle=shuffle,
    #         path=DATASET_ROOT/'beijing_pm25',
    #         batch_size=batch_size,
    #     )
    
    # # Compute number of batches and steps for learning rate scheduler.
    # batches = tf.data.experimental.cardinality(dataset_train).numpy()
    # n_steps = epochs*batches

    # Build the parameter grid.
    grid = ParameterGrid(params)
    n_grid = len(grid)
    logger.info(f"[{model_name}] Evaluating {n_grid} hyperparameter combinations")

    # Save parameter grid to file.
    with open(tuning_model_root/'parameter_grid.json', 'w') as f:
        json.dump(list(grid), f, default=lambda o: '<not serializable>')

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

                # Load the dataset.
                dataset_train, dataset_val, dataset_test = dataset_loader_func(batch_size=batch_size)

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
                metric_list=metric_list,
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

    logger.info(f"[{model_name}] Tuning Results:")

    # Build dataframe using results.
    df = pd.DataFrame(df_results)
    if table_header is not None:
        df = df[table_header]
    if sort_cols is not None:
        df = df.sort_values(by=sort_cols, ascending=sort_ascending)
    logger.info(df.to_string(index=False)) # Log to console.
    # df.to_csv(TABLE_ROOT/f"{model_name}_tuning_results.csv", sep='|', index=False)

    # # Export the dataframe to LaTeX using custom style.
    # df = df.drop(columns=table_omit_cols, errors='ignore')
    # df.columns = df.columns.map(lambda x: x.replace('_', '\_')) # Escape the header names too.
    # styler = df.style
    # styler = styler.format(str, escape='latex') # Default is to convert all cells to their string representation.
    # subset = []
    # for m in compile_params['metrics']+['loss']:
    #     if m in set(df.columns):
    #         subset.append(f"{m}")
    #         subset.append(f"val\_{m}")
    #         subset.append(f"test\_{m}")
    # styler = styler.format(formatter='{:.4f}', subset=subset)
    # styler = styler.highlight_min(subset=subset, axis=0, props='textbf:--rwrap;')
    # styler = styler.hide(axis=0) # Hide the index.
    # styler.to_latex(
    #     buf=TABLE_ROOT/f"{model_name}_tuning_results.tex",
    #     hrules=True,
    # )
    # df.columns = df.columns.map(lambda x: x.replace('\_', '_')) # Convert header names back.

    # # List of colors for plotting.
    # n = len(histories)
    # color = plt.cm.rainbow(np.linspace(0, 1, n))


    # # Plot train/val performance.
    # for key in compile_params['metrics']+['loss']:

    #     fig = plt.figure(figsize=(8,6))
    #     for i, (h, c) in enumerate(zip(histories, color)):
    #         plt.plot(h[key], label=f"model {i} train", color=c, linestyle='-')
    #         plt.xlim(0, len(h[key])-1)
    #     plt.xlabel('Epoch')
    #     plt.ylabel(key.upper())
    #     plt.title(f"{model_name.upper()} Training {key.upper()}")
    #     plt.legend(loc='center left', ncol=2, bbox_to_anchor=(1.04,0.5))
    #     fig.savefig(IMAGE_ROOT/f"{model_name}_hp_{key}_train.png", bbox_inches='tight')
    #     fig.show()

    #     fig = plt.figure(figsize=(8,6))
    #     for i, (h, c) in enumerate(zip(histories, color)):
    #         plt.plot(h[f'val_{key}'], label=f"model {i} val", color=c, linestyle='-')
    #         plt.xlim(0, len(h[key])-1)
    #     plt.xlabel('Epoch')
    #     plt.ylabel(key.upper())
    #     plt.title(f"{model_name.upper()} Validation {key.upper()}")
    #     plt.legend(loc='center left', ncol=2, bbox_to_anchor=(1.04,0.5))
    #     fig.savefig(IMAGE_ROOT/f"{model_name}_hp_{key}_val.png", bbox_inches='tight')
    #     fig.show()

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

    # # Plot the best model predictions of the dataset.
    # figs: dict = plot_predictions(
    #     model=model,
    #     in_seq_len=in_seq_len,
    #     out_seq_len=out_seq_len,
    #     shift=shift,
    #     split=split,
    #     in_feat=in_feat,
    #     out_feat=out_feat,
    #     batch_size=batch_size,
    #     shuffle=shuffle,
    #     strategy=strategy,
    # )
    # for name, fig in figs.items():
    #     fig.savefig(IMAGE_ROOT/f"{model_name}_io_{name}.png", bbox_inches='tight')
    #     fig.show()

    return model, hist, met, params, df