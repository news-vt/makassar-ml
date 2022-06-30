from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt


def LinearWarmupLearningRateScheduleWrapper(base):
    """Learning rate schedule wrapper with linear warmup.

    This function creates instances of `tf.keras.optimizers.schedules.LearningRateSchedule` which have a linear warmup prior to their normal function.
    """
    # Ensure base class is a subclass of `LearningRateSchedule`.
    assert isinstance(base, Callable)
    assert issubclass(base, tf.keras.optimizers.schedules.LearningRateSchedule)

    # Wrapper class.
    name = f"LinearWarmup{base.__name__}"
    class Wrapper(base):
        def __init__(self, 
            warmup_learning_rate: float,
            warmup_steps: int,
            **kwargs,
            ):
            super().__init__(**kwargs)
            self.warmup_learning_rate = warmup_learning_rate
            self.warmup_steps = warmup_steps

            # Compute offset from base class initial value and the final warmup learning rate.
            self._step_size = super(Wrapper, self).__call__(0) - self.warmup_learning_rate

        def __call__(self, step: int):
            learning_rate = tf.cond(
                pred=tf.less(step, self.warmup_steps), # If steps < warmup_steps
                true_fn=lambda: (
                    self.warmup_learning_rate + tf.cast(step, dtype=tf.float32)/self.warmup_steps*self._step_size
                ),
                false_fn=lambda: (super(Wrapper, self).__call__(step)),
            )
            return learning_rate

        def get_config(self) -> dict:
            config = super().get_config().copy()
            config.update({
                'warmup_learning_rate': self.warmup_learning_rate,
                'warmup_steps': self.warmup_steps,
            })
            return config

    # Override the wrapper class name as seen by Python.
    Wrapper.__name__ = name

    # Add this new custom wrapper to the list of custom objects.
    keras.utils.get_custom_objects()[Wrapper.__name__] = Wrapper

    # Return the wrapper.
    return Wrapper


def LinearWarmupCosineDecay(*args, **kwargs):
    """Cosine Decay with linear warmup."""
    return LinearWarmupLearningRateScheduleWrapper(
        keras.experimental.CosineDecay
        )(*args, **kwargs)




class LearningRateDecay:
    def __call__(self, epoch: int):
        raise NotImplementedError

    def plot(self, epochs: list[float]):
        epochs = np.array(epochs)
        lrs = self(epochs)
        plt.plot(epochs, lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')


class CosineDecay(LearningRateDecay):
    """Cosine decay over epochs.

    Implementation inspired by https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay.
    """
    def __init__(self, 
        init_lr: float,
        decay_epochs: int,
        alpha: float = 0.0,
        ):
        super().__init__()
        self.init_lr = init_lr
        self.decay_epochs = decay_epochs
        self.alpha = alpha

    def __call__(self, epoch: int):
        epoch = min(epoch, self.decay_epochs)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * epoch / self.decay_epochs))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.init_lr * decayed


class LinearWarmupLinearDecay(LearningRateDecay):
    def __init__(self,
        warmup_epochs: int,
        decay_epochs: int,
        initial_lr: float,
        base_lr: float,
        min_lr: float,
        ):
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.initial_lr = initial_lr
        self.base_lr = base_lr
        self.min_lr = min_lr

    def __call__(self, epoch: int):
        if epoch <= self.warmup_epochs:
            pct = epoch / self.warmup_epochs
            return ((self.base_lr - self.initial_lr) * pct) + self.initial_lr

        if epoch > self.warmup_epochs and epoch < self.warmup_epochs+self.decay_epochs:
            pct = 1 - ((epoch - self.warmup_epochs) / self.decay_epochs)
            return ((self.base_lr - self.min_lr) * pct) + self.min_lr

        return self.min_lr


def lr_scheduler_linear_warmup_linear_decay(
    epoch: int, # Current epoch
    lr: float, # Current learning rate
    warmup_epochs: int = 15,
    decay_epochs: int = 100,
    initial_lr: float = 1e-6,
    base_lr: float = 1e-3,
    min_lr: float = 5e-5,
    ):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr