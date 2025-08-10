from typing import Sequence

import matplotlib.pyplot as plt


class Visualizer:
    @staticmethod
    def plot_accuracy_and_loss(fold_accuracies: Sequence[float], fold_losses: Sequence[float]) -> None:
        folds = range(1, len(fold_accuracies) + 1)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(folds, fold_accuracies, marker="o", linestyle="-", color="b", label="Validation Accuracy")
        plt.xlabel("Fold")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy per Fold")
        plt.xticks(folds)
        plt.legend()
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(folds, fold_losses, marker="o", linestyle="-", color="r", label="Validation Loss")
        plt.xlabel("Fold")
        plt.ylabel("Loss")
        plt.title("Validation Loss per Fold")
        plt.xticks(folds)
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_prf(fold_precisions: Sequence[float], fold_recalls: Sequence[float], fold_f1_scores: Sequence[float]) -> None:
        folds = range(1, len(fold_precisions) + 1)
        plt.figure(figsize=(15, 5))

        for i, (metric, title, color) in enumerate(
            zip([fold_precisions, fold_recalls, fold_f1_scores], ["Precision", "Recall", "F1-score"], ["g", "c", "m"])
        ):
            plt.subplot(1, 3, i + 1)
            plt.plot(folds, metric, marker="o", linestyle="-", color=color)
            plt.xlabel("Fold")
            plt.ylabel("Score")
            plt.title(f"{title} per Fold")
            plt.xticks(folds)
            plt.grid()

        plt.tight_layout()
        plt.show()


