from __future__ import annotations
import tensorflow.keras as keras


class LearningRateAdjuster(keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        self.original_lr = self.model.optimizer.lr.read_value()
        return super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        new_lr = self.original_lr * (0.5 ** ((epoch-1) // 1))
        self.model.optimizer.lr.assign(new_lr)
        return super().on_epoch_end(epoch, logs)