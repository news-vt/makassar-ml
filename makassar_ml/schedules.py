from __future__ import annotations
import tensorflow as tf
import tensorflow.keras as keras
from typing import Callable

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