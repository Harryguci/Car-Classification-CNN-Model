from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir: str = "./vehicle_dataset/train"
    img_height: int = 256
    img_width: int = 256
    batch_size: int = 16
    num_classes: int = 3
    k_folds: int = 5
    learning_rate: float = 1e-3
    epochs: int = 50
    early_stopping_patience: int = 5
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 3
    reduce_lr_min_lr: float = 3e-4
    l2_reg: float = 1e-3
    dropout_rate: float = 0.5
    model_checkpoint_path: str = "./vehicle_best_model.keras"


