from __future__ import annotations
import inspect
import json
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


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