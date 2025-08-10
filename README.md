# Vehicle Classification Project

A deep learning-based vehicle classification system that can classify images into three categories: **car**, **truck**, and **bus**. This project implements a CNN-based classifier with advanced training techniques including k-fold cross-validation, data augmentation, and comprehensive evaluation metrics.

## 🚗 Features

- **Multi-class Classification**: Classifies vehicles into car, truck, and bus categories
- **Deep Learning Model**: CNN architecture with data augmentation and regularization
- **Advanced Training**: K-fold cross-validation with early stopping and learning rate scheduling
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, and loss metrics
- **Data Management**: Automatic dataset integrity checking and corrupt image removal
- **Visualization**: Training progress plots and performance metrics visualization
- **GPU Support**: Optimized for GPU training with memory growth management
- **Prediction Interface**: Easy-to-use prediction API for new images

## 🏗️ Architecture

The project uses a **Convolutional Neural Network (CNN)** with the following architecture:

- **Input Layer**: 256x256 RGB images
- **Data Augmentation**: Random flip, rotation, and zoom
- **Convolutional Layers**: 4 layers with increasing filters (16→32→64→128)
- **Regularization**: Batch normalization, dropout (50%), and L2 regularization
- **Pooling**: MaxPooling2D after each convolutional layer
- **Dense Layers**: 256 neurons with ReLU activation
- **Output Layer**: 3 neurons with softmax activation

## 📁 Project Structure

```
car_classification/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                     # Source code
│   ├── __init__.py
│   ├── main.py             # Main training script
│   ├── config.py           # Configuration settings
│   ├── data.py             # Dataset management
│   ├── model.py            # CNN model definition
│   ├── trainer.py          # Training logic
│   ├── predictor.py        # Prediction interface
│   ├── visualization.py    # Plotting utilities
│   ├── test.py            # Testing utilities
│   └── utils/
│       └── gpu.py         # GPU utilities
└── vehicle_dataset/        # Dataset directory
    ├── train/             # Training data
    │   ├── bus/
    │   ├── car/
    │   └── truck/
    └── test/              # Test data
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- TensorFlow 2.20+

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd car_classification
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

The project expects a dataset organized as follows:
- **Training data**: `vehicle_dataset/train/` with subdirectories for each class
- **Test data**: `vehicle_dataset/test/` with subdirectories for each class
- **Supported classes**: `bus/`, `car/`, `truck/`
- **Image format**: JPG, PNG, or other common image formats
- **Image size**: Automatically resized to 256x256 pixels

## 🎯 Usage

### Training

Run the main training script:

```bash
python -m src.main
```

This will:
- Load and preprocess the dataset
- Train the model using 5-fold cross-validation
- Display training progress and results
- Save the best model checkpoint
- Generate performance visualizations

### Configuration

Modify training parameters in `src/config.py`:

```python
config = TrainingConfig(
    data_dir="./vehicle_dataset/train",
    img_height=256,
    img_width=256,
    batch_size=16,
    num_classes=3,
    k_folds=5,
    learning_rate=1e-3,
    epochs=50,
    # ... other parameters
)
```

### Prediction

Use the trained model to classify new images:

```python
from src.predictor import Predictor

predictor = Predictor("./vehicle_best_model.keras")
result = predictor.predict("path/to/your/image.jpg", show=True)
print(f"Predicted class: {result}")
```

## 📈 Training Features

### K-Fold Cross-Validation
- **5-fold cross-validation** for robust model evaluation
- **Mean and standard deviation** of all metrics across folds

### Data Augmentation
- **Random horizontal flip**
- **Random rotation** (±10%)
- **Random zoom** (±10%)

### Regularization Techniques
- **L2 regularization** (1e-3) on convolutional layers
- **Dropout** (50%) on dense layers
- **Batch normalization** after each convolutional layer

### Training Optimizations
- **Early stopping** with patience of 5 epochs
- **Learning rate reduction** with patience of 3 epochs
- **Adam optimizer** with configurable learning rate

## 📊 Evaluation Metrics

The model provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Loss**: Training and validation loss

## 🖥️ GPU Support

The project automatically enables GPU memory growth to prevent memory issues:

```python
from src.utils.gpu import enable_gpu_memory_growth
enable_gpu_memory_growth()
```

## 🔧 Customization

### Adding New Classes
1. Add new class directories to the dataset
2. Update `num_classes` in the configuration
3. Retrain the model

### Model Architecture
Modify the CNN architecture in `src/model.py`:
- Add/remove convolutional layers
- Change filter sizes and counts
- Adjust dense layer dimensions

### Training Parameters
Customize training behavior in `src/config.py`:
- Learning rate and scheduling
- Batch size and epochs
- Regularization strength
- Early stopping patience

## 📝 Requirements

Key dependencies:
- **TensorFlow**: 2.20+ (deep learning framework)
- **Keras**: 3.11+ (high-level neural network API)
- **NumPy**: 2.3+ (numerical computing)
- **Matplotlib**: 3.10+ (visualization)
- **Scikit-learn**: 1.7+ (machine learning utilities)
- **Pillow**: 11.3+ (image processing)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Contributors to the open-source machine learning ecosystem

---

**Note**: This project is designed for educational and research purposes. For production use, consider additional validation, testing, and deployment considerations.
