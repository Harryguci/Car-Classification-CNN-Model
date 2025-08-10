"""Vehicle Classification package.

Provides OOP interfaces for data loading, model building, training with K-Fold
cross validation, visualization, and prediction.
"""

from .config import TrainingConfig
from .data import DatasetManager
from .model import VehicleClassifierModel
from .trainer import Trainer, TrainingResults
from .visualization import Visualizer
from .predictor import Predictor
from .utils.gpu import enable_gpu_memory_growth

__all__ = [
    "TrainingConfig",
    "DatasetManager",
    "VehicleClassifierModel",
    "Trainer",
    "TrainingResults",
    "Visualizer",
    "Predictor",
    "enable_gpu_memory_growth",
]


