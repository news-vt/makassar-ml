from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


def plot_metric(history: dict, metric: str) -> plt.Figure:
    """Plots model training and validation metric."""
    fig = plt.figure()
    plt.plot(history[metric], label='train')
    plt.plot(history[f"val_{metric}"], label='val')
    plt.xlim(0, len(history[metric])-1)
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.legend(loc='upper left')
    return fig


def plot_feature_partitions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    x_key: str,
    figsize: list[int] = (15,15),
    ) -> plt.Figure:
    """Plots features within train/val/test dataframes."""
    fig, axs = plt.subplots(nrows=len(features), figsize=figsize)
    for i,key in enumerate(features):
        sns.lineplot(data=train_df, x=x_key, y=key, label='train', ax=axs[i])
        sns.lineplot(data=val_df, x=x_key, y=key, label='val', ax=axs[i])
        sns.lineplot(data=test_df, x=x_key, y=key, label='test', ax=axs[i])
        axs[i].legend(loc='upper right')
    return fig


def plot_input_output(
    df: pd.DataFrame,
    pred: tf.Tensor,
    in_seq_len: int,
    out_seq_len: int,
    shift: int,
    in_feat: list[str],
    out_feat: list[str],
    x_key: str,
    figsize: tuple = (15,7),
    width_ratios: list = [1, 1.75],
    wspace: float = 0.07,
    xtick_rotation: float = 45,
    ) -> plt.Figure:
    """Creates figure for input/output feature data."""
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.subfigures(nrows=1, ncols=2, wspace=wspace, width_ratios=width_ratios)

    # Plot all input features.
    subfigs[0].suptitle('Input Features', fontsize='x-large')
    axs = subfigs[0].subplots(nrows=len(in_feat), ncols=1, sharex=True, squeeze=False)
    for r, key in enumerate(in_feat):
        sns.lineplot(
            data=df,
            x=x_key,
            y=key,
            ax=axs[r,0],
        )
    plt.xticks(rotation=xtick_rotation)

    # Plot all output features against truth.
    subfigs[1].suptitle('Output Features', fontsize='x-large')
    axs = subfigs[1].subplots(nrows=len(out_feat), ncols=1, sharex=True, squeeze=False)
    # Predictions are batched, so flatten them for plotting.
    pred = tf.reshape(pred, shape=(-1, pred.shape[-1]))
    for r, key in enumerate(out_feat):
        sns.lineplot(
            data=df,
            x=x_key,
            y=key,
            ax=axs[r,0],
            label='truth',
        )
        axs[r,0].plot(
            df.iloc[
                np.arange(
                    start=in_seq_len,
                    stop=pred.shape[0]+in_seq_len,
                    step=shift,
                )
            ][x_key],
            pred[:,r],
            linewidth=3,
            label=f'forecast',
        )
        axs[r,0].legend(loc='upper right')
    plt.xticks(rotation=xtick_rotation)

    return fig