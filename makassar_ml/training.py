from __future__ import annotations
import inspect
import json
import logging
from typing import Callable
import pandas as pd
from pathlib import Path
import tensorflow as tf
import tensorflow.keras as keras

# Create logger for module.
logger = logging.getLogger(__name__)


def load_metrics(metrics_path: str):
    """Load model metrics from file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def load_history(history_path: str):
    """Load model history from file."""
    return pd.read_csv(history_path)


def load_trained_model(
    checkpoint_path: str,
    ) -> tuple[keras.models.Model]:
    """Helper to load a saved model."""
    model = keras.models.load_model(
        checkpoint_path, 
        custom_objects=keras.utils.get_custom_objects(),
    )
    return model


def build_model_from_hparams(func):
    """Generalized model build and compile from hyperparameters."""
    def wrapper(hparams: dict, compile_params: dict) -> keras.Model:
        """Builds model using given hyperparameters for both model and optimizer, and compile parameters for compilation."""
        # Extract paramters needed for the model.
        model_params = {k: hparams[k] for k in inspect.signature(func).parameters if k in hparams}
        # Build model.
        model = func(**model_params)
        # Configure optimizer.
        optim = keras.optimizers.get({
            'class_name': hparams['optim'],
            'config': {
                'lr': hparams['lr'],
            },
        })
        # Compile the model.
        model.compile(optimizer=optim, loss=compile_params['loss'], metrics=compile_params['metrics'])
        return model
    return wrapper


def train_evaluate_model(
    model,
    datagen_train: tf.data.Dataset,
    datagen_val: tf.data.Dataset,
    datagen_test: tf.data.Dataset,
    epochs: int,
    metric_list: list[str],
    checkpoint_path: str,
    history_path: str = None,
    metrics_path: str = None,
    ) -> tuple[keras.models.Model, dict, dict]:
    """Trains and evaluates a given model on the given datasets.

    Args:
        model (_type_): The model to train and evaluation.
        datagen_train (tf.data.Dataset): Training dataset.
        datagen_val (tf.data.Dataset): Validation dataset.
        datagen_test (tf.data.Dataset): Testing dataset.
        epochs (int): Number of training epochs.
        checkpoint_path (str): Path to checkpoint file
        history_path (str, optional): Path to history CSV file. If None is provided, then the file will be located at the same path as the checkpoint file with name `"history.csv"`. Defaults to None.
        metrics_path (str, optional): Path to metrics JSON file. If None is provided, then the file will be located at the same path as the checkpoint file with name `"metrics.json"`. Defaults to None.

    Returns:
        tuple[keras.models.Model, dict, dict: Tuple of trained model, history dictionary, and metrics dictionary.
    """

    # Ensure checkpoint root directory has been created.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path is None:
        history_path = checkpoint_path.parent/'history.csv'
    if metrics_path is None:
        metrics_path = checkpoint_path.parent/'metrics.json'

    # List of callbacks during training.
    callbacks = [
        # Save model checkpoint after every epoch.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1,
        ),
        # Log training history to CSV file.
        keras.callbacks.CSVLogger(
            filename=history_path,
            append=False,
        ),
        # Early stopping when performance does not improve across N epochs.
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='auto',
            patience=10,
            # min_delta=0.001,
            restore_best_weights=True,
        ),
    ]

    # Train the model.
    history = model.fit(datagen_train,
        validation_data=datagen_val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate the newly trained model.
    test_metrics = model.evaluate(datagen_test)

    # Create dictionary of metrics to return and preserve in file.
    metrics = {}
    for i, key in enumerate(metric_list):
        metrics[key] = history.history[key][-1]
        metrics[f"val_{key}"] = history.history[f"val_{key}"][-1]
        metrics[f"test_{key}"] = test_metrics[i+1]
    metrics['loss'] = history.history['loss'][-1]
    metrics['val_loss'] = history.history['val_loss'][-1]
    metrics['test_loss'] = test_metrics[0]

    # Dump metrics to JSON file.
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    return model, history.history, metrics


def ensure_path(s: None|str|Path, default: str|Path = '.') -> Path:
    if s is None:
        s = Path(default)
    elif not isinstance(s, Path):
        s = Path(s)
    return s


def train_evaluate_for_dataset(
    model_name: str,
    build_model_func: Callable[[], keras.Model],
    dataset_loader_func: Callable[[int], tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]],
    metric_list: list[str],
    batch_size: int = 128,
    strategy: tf.distribute.Strategy = tf.distribute.get_strategy(),
    epochs: int = 10,
    checkpoint_root: str = None,
    ) -> tuple[keras.Model, dict, dict]:
    """Train and evaluate a model on a given dataset.

    If checkpoint exists then the model is loaded in place of training.
    """
    # Ensure roots are path objects.
    checkpoint_root = ensure_path(checkpoint_root)

    # Train and evaluate model.
    checkpoint_path = checkpoint_root/model_name/'model.h5'
    history_path = checkpoint_path.parent/'history.csv'
    metrics_path = checkpoint_path.parent/'metrics.json'

    # Maximize batch size efficiency using distributed strategy.
    batch_size_per_replica = batch_size
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

    # Load model from best checkpoint.
    if checkpoint_path.exists() and history_path.exists() and metrics_path.exists():
        # Load the model.
        logger.info(f"[{model_name}] Loading best model from: {checkpoint_path}")
        with strategy.scope():
            model = keras.models.load_model(checkpoint_path, custom_objects=keras.utils.get_custom_objects())

        # Load history and metrics.
        logger.info(f"[{model_name}] Loading from save data")
        hist = load_history(history_path)
        met = load_metrics(metrics_path)

    # Train model.
    else:
        logger.info(f"[{model_name}] Training new model: {epochs=}, {batch_size=}, {strategy=}")

        # Use strategy memory.
        with strategy.scope():

            # Load the dataset.
            dataset_train, dataset_val, dataset_test = dataset_loader_func(batch_size=batch_size)

            # Create and compile model.
            model = build_model_func()

        # Train the model using the strategy.
        model, hist, met = train_evaluate_model(
            model,
            datagen_train=dataset_train,
            datagen_val=dataset_val,
            datagen_test=dataset_test,
            epochs=epochs,
            metric_list=metric_list,
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            metrics_path=metrics_path,
        )

    return model, hist, met


    # # Plot the model predictions of the dataset.
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
    #     fig.savefig(image_root/f"{model_name}_io_{name}.png", bbox_inches='tight')
    #     fig.show()

    # # Print model summary.
    # logger.info(f"[{model_name}] Model Summary:")
    # model.summary(print_fn=logger.info)

    # logger.info(f"[{model_name}] Training and Evaluation Results:")

    # # Build dataframe using results.
    # df = pd.DataFrame([{
    #     'model': model_name,
    #     **met,
    #     **params,
    # }])
    # if table_header is not None:
    #     df = df[table_header]
    # df = df.drop(columns=table_omit_cols, errors='ignore')
    # logger.info(df.to_string(index=False)) # Log to console.

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
    #     buf=table_root/f"{model_name}_results.tex",
    #     hrules=True,
    # )
    # df.columns = df.columns.map(lambda x: x.replace('\_', '_')) # Convert header names back.

    # # Plot train/val performance.
    # for key in compile_params['metrics']+['loss']:
    #     fig = plot_metric(hist, key)
    #     fig.savefig(image_root/f"{model_name}_{key}.png", bbox_inches='tight')
    #     fig.show()

    # return model, hist, met, params, df