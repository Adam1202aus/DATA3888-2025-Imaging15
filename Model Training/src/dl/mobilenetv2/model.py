# src/dl/mobilenetv2/model.py
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


def build_mobilenetv2_model(num_classes: int,
                            input_shape: tuple[int, int, int] = (224, 224, 3),
                            base_trainable: bool | int = False,
                            dropout: float = 0.3) -> keras.Model:
    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    if isinstance(base_trainable, bool):
        base.trainable = base_trainable
        print(f"  MobileNetV2 base.trainable set to: {base_trainable}")
    elif isinstance(base_trainable, int) and base_trainable >= 0:
        if base_trainable == 0:
            print(f"  Freezing all layers of MobileNetV2 base.")
            base.trainable = False
        elif len(base.layers) >= base_trainable:
            print(f"  Unfreezing last {base_trainable} layers of MobileNetV2 base.")
            for layer in base.layers:
                layer.trainable = False
            for layer in base.layers[-base_trainable:]:
                layer.trainable = True
        else:
            print(
                f"  Warning: base_trainable_setting ({base_trainable}) >= num layers in MobileNetV2 base ({len(base.layers)}). Unfreezing all base layers.")
            base.trainable = True
    else:
        print(f"  Invalid base_trainable_setting: {base_trainable}. Freezing all layers of MobileNetV2 base.")
        base.trainable = False


    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    is_any_layer_in_base_trainable = any(layer.trainable for layer in base.layers)
    x_base = base(x, training=is_any_layer_in_base_trainable)

    if 0 < dropout < 1:
        x_final = layers.Dropout(dropout)(x_base)
    else:
        x_final = x_base

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x_final)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CellTypeMobileNetV2")
    return model


__all__ = ["build_mobilenetv2_model"]