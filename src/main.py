from . import (
    TrainingConfig,
    DatasetManager,
    VehicleClassifierModel,
    Trainer,
    Visualizer,
    Predictor,
    enable_gpu_memory_growth,
)


def main() -> None:
    enable_gpu_memory_growth()

    from pathlib import Path
    project_root = Path(__file__).resolve().parents[1]
    resolved_data_dir = str(project_root / "vehicle_dataset" / "train")

    config = TrainingConfig(
        data_dir=resolved_data_dir,
        img_height=256,
        img_width=256,
        batch_size=16,
        num_classes=3,
        k_folds=5,
        learning_rate=1e-3,
        epochs=50,
        early_stopping_patience=5,
        reduce_lr_factor=0.5,
        reduce_lr_patience=3,
        reduce_lr_min_lr=3e-4,
        l2_reg=1e-3,
        dropout_rate=0.5,
        model_checkpoint_path="./vehicle_best_model.keras",
    )

    dataset_manager = DatasetManager(config)
    dataset_manager.remove_corrupt_images()
    print("Dataset integrity check complete.")

    X, y = dataset_manager.load_dataset_arrays()
    print(f"Dataset loaded: {X.shape}, Labels: {y.shape}")

    VehicleClassifierModel(config).build().summary()

    trainer = Trainer(config)
    results = trainer.train_kfold(X, y)

    import numpy as np

    print("\nK-Fold Cross Validation Results")
    print(f"Mean Accuracy: {np.mean(results.fold_accuracies) * 100:.2f}% (+/- {np.std(results.fold_accuracies) * 100:.2f}%)")
    print(f"Mean Loss: {np.mean(results.fold_losses)*100:.4f} (+/- {np.std(results.fold_losses)*100:.4f})")
    print(f"Mean Precision: {np.mean(results.fold_precisions)*100:.4f} (+/- {np.std(results.fold_precisions)*100:.4f})")
    print(f"Mean Recall: {np.mean(results.fold_recalls)*100:.4f} (+/- {np.std(results.fold_recalls)*100:.4f})")
    print(f"Mean F1-score: {np.mean(results.fold_f1_scores)*100:.4f} (+/- {np.std(results.fold_f1_scores)*100:.4f})")

    Visualizer.plot_accuracy_and_loss(results.fold_accuracies, results.fold_losses)
    Visualizer.plot_prf(results.fold_precisions, results.fold_recalls, results.fold_f1_scores)

    # Example usage for local prediction
    # predictor = Predictor(config.model_checkpoint_path)
    # predictor.predict("path/to/your/image.jpg", show=True)


if __name__ == "__main__":
    main()


