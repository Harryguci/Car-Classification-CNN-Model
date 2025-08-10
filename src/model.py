from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

from .config import TrainingConfig


class VehicleClassifierModel:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    # Data augmentation
    def _augmentation(self) -> tf.keras.Sequential:
        return tf.keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
            ]
        )

    def build(self) -> tf.keras.Model:
        c = self.config
        model = models.Sequential(
            [
                layers.Input(shape=(c.img_height, c.img_width, 3)),
                self._augmentation(),
                layers.Conv2D(16, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(c.l2_reg)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(c.l2_reg)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(c.l2_reg)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(c.l2_reg)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(self.config.dropout_rate),
                layers.Dense(c.num_classes, activation="softmax", dtype=tf.float32),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=c.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


