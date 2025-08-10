from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from .config import TrainingConfig
from .model import VehicleClassifierModel


@dataclass
class TrainingResults:
    fold_accuracies: List[float]
    fold_losses: List[float]
    fold_precisions: List[float]
    fold_recalls: List[float]
    fold_f1_scores: List[float]
    histories: List[tf.keras.callbacks.History]


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def _callbacks(self) -> list:
        return [
            EarlyStopping(monitor="val_loss", patience=self.config.early_stopping_patience, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=self.config.reduce_lr_min_lr,
            ),
            ModelCheckpoint(
                self.config.model_checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
        ]

    def train_kfold(self, X: np.ndarray, y: np.ndarray) -> TrainingResults:
        kfold = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=1301)

        fold_accuracies: list[float] = []
        fold_losses: list[float] = []
        fold_precisions: list[float] = []
        fold_recalls: list[float] = []
        fold_f1_scores: list[float] = []
        histories: list[tf.keras.callbacks.History] = []

        for fold_index, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\nTraining Fold {fold_index + 1}/{self.config.k_folds}...")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_ds = (
                tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .batch(self.config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            val_ds = (
                tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(self.config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            model = VehicleClassifierModel(self.config).build()

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.epochs,
                callbacks=self._callbacks(),
                verbose=1,
            )
            histories.append(history)

            y_pred_probs = model.predict(X_val)
            y_pred = np.argmax(y_pred_probs, axis=1)

            val_accuracy = float(np.mean(y_pred == y_val))
            val_precision = float(precision_score(y_val, y_pred, average="macro"))
            val_recall = float(recall_score(y_val, y_pred, average="macro"))
            val_f1 = float(f1_score(y_val, y_pred, average="macro"))

            print(f"Fold {fold_index + 1} - Acc: {val_accuracy:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")

            fold_accuracies.append(val_accuracy)
            fold_losses.append(float(history.history["val_loss"][ -1 ]))
            fold_precisions.append(val_precision)
            fold_recalls.append(val_recall)
            fold_f1_scores.append(val_f1)

        return TrainingResults(
            fold_accuracies=fold_accuracies,
            fold_losses=fold_losses,
            fold_precisions=fold_precisions,
            fold_recalls=fold_recalls,
            fold_f1_scores=fold_f1_scores,
            histories=histories,
        )


