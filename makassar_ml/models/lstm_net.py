from __future__ import annotations
import tensorflow.keras as keras

def build_model(
    in_seq_len: int,
    in_feat: int,
    out_feat: int,
    lstm_units: list[int], # list of LSTM dimensions.
    fc_units: list[int], # list of fully-connected dimensions.
    dropout: float = 0.0,
    ) -> keras.Model:
    inp = keras.Input(shape=(in_seq_len, in_feat))

    # Build LSTM layers.
    x = inp
    for i, units in enumerate(lstm_units):
        x = keras.layers.LSTM(
            units=units,
            return_sequences=(i<len(lstm_units)-1),
        )(x)

    # Build fully-connected layers.
    for units in fc_units:
        x = keras.layers.Dense(units=units, activation='linear')(x)
        x = keras.layers.Dropout(rate=dropout)(x)

    # Classifier.
    x = keras.layers.Dense(units=out_feat, activation='linear')(x)

    # Construct model class and return.
    return keras.Model(inputs=inp, outputs=x)