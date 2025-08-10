# src/services/ml_model_service.py
import tensorflow as tf
import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger(__name__)

class MLModelService:
    """Service for ML model operations: load, predict, save, fine-tune"""
    
    def __init__(self, model_path: str = "models/vehicle_classifier.keras"):
        self.model_path = model_path
        self.model = None
        self.class_names = ["car", "bus", "truck"]
        self.input_shape = (224, 224, 3)
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
    
    # Feature 1: Load the model
    def load_model(self, model_path: str = None) -> bool:
        """Load a .keras model file"""
        try:
            path = model_path or self.model_path
            
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            self.model = tf.keras.models.load_model(path)
            self.model_path = path
            logger.info(f"Model loaded successfully from {path}")
            
            # Update input shape from loaded model
            if self.model.input_shape:
                self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    # Feature 2: Predict the class of the image
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Predict the class of a single image"""
        try:
            if self.model is None:
                raise ValueError("No model loaded. Please load a model first.")
            
            # Load and preprocess image
            image = self._preprocess_image(image_path)
            if image is None:
                raise ValueError("Failed to load or preprocess image")
            
            # Make prediction
            image_batch = np.expand_dims(image, axis=0)
            predictions = self.model.predict(image_batch, verbose=0)
            
            # Get results
            class_probabilities = predictions[0]
            predicted_class_idx = np.argmax(class_probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(class_probabilities[predicted_class_idx])
            
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, class_probabilities)
                }
            }
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error predicting image: {e}")
            return {"error": str(e)}
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Predict classes for multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict_image(image_path)
            result["image_path"] = image_path
            results.append(result)
        return results
    
    # Feature 3: Save the model
    def save_model(self, save_path: str = None) -> bool:
        """Save the current model as .keras file"""
        try:
            if self.model is None:
                logger.error("No model to save")
                return False
            
            path = save_path or self.model_path
            
            # Ensure .keras extension
            if not path.endswith('.keras'):
                path += '.keras'
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            self.model.save(path)
            logger.info(f"Model saved successfully to {path}")
            
            # Update current model path
            self.model_path = path
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    # Feature 4: Fine-tune the model
    def fine_tune_model(self, 
                       training_data_dir: str,
                       epochs: int = 10,
                       batch_size: int = 32,
                       learning_rate: float = 0.0001,
                       validation_split: float = 0.2,
                       freeze_layers: int = None,
                       save_best_model: bool = True) -> Dict[str, Any]:
        """Fine-tune the loaded model with new data"""
        try:
            if self.model is None:
                raise ValueError("No model loaded. Please load a model first.")
            
            # Load training data
            X, y = self._load_training_data(training_data_dir)
            if X is None or len(X) == 0:
                raise ValueError("No training data found")
            
            logger.info(f"Loaded {len(X)} training samples")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, 
                stratify=y.argmax(axis=1) if len(y.shape) > 1 else y
            )
            
            # Prepare model for fine-tuning
            self._prepare_for_finetuning(learning_rate, freeze_layers)
            
            # Create data generators with augmentation
            train_gen, val_gen = self._create_data_generators(
                X_train, y_train, X_val, y_val, batch_size
            )
            
            # Setup callbacks
            callbacks = self._setup_callbacks(save_best_model)
            
            # Train the model
            logger.info("Starting fine-tuning...")
            history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=callbacks,
                verbose=1
            )
            
            # Save the fine-tuned model
            if save_best_model:
                self.save_model()
            
            # Return results
            return {
                "status": "success",
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "epochs_completed": len(history.history['loss']),
                "final_accuracy": float(history.history.get('accuracy', [0])[-1]),
                "final_val_accuracy": float(history.history.get('val_accuracy', [0])[-1]),
                "final_loss": float(history.history.get('loss', [0])[-1]),
                "final_val_loss": float(history.history.get('val_loss', [0])[-1]),
                "model_path": self.model_path
            }
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return {"status": "error", "error": str(e)}
    
    # Helper methods
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for model input"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                # Try with PIL if cv2 fails
                image = Image.open(image_path)
                image = np.array(image)
            
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            target_size = (self.input_shape[1], self.input_shape[0])  # (width, height)
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _load_training_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from directory structure"""
        try:
            images = []
            labels = []
            
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(data_dir, class_name)
                
                if not os.path.exists(class_dir):
                    logger.warning(f"Class directory not found: {class_dir}")
                    continue
                
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                for image_file in image_files:
                    image_path = os.path.join(class_dir, image_file)
                    image = self._preprocess_image(image_path)
                    
                    if image is not None:
                        images.append(image)
                        # Create one-hot encoded label
                        label = np.zeros(len(self.class_names))
                        label[class_idx] = 1
                        labels.append(label)
                
                logger.info(f"Loaded {len(image_files)} images from {class_name}")
            
            return np.array(images), np.array(labels)
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None, None
    
    def _prepare_for_finetuning(self, learning_rate: float, freeze_layers: int = None):
        """Prepare model for fine-tuning"""
        # Freeze layers if specified
        if freeze_layers is not None:
            for i, layer in enumerate(self.model.layers):
                layer.trainable = i >= freeze_layers
            logger.info(f"Frozen first {freeze_layers} layers")
        
        # Compile model with new learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _create_data_generators(self, X_train, y_train, X_val, y_val, batch_size):
        """Create data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def _setup_callbacks(self, save_best_model: bool) -> List:
        """Setup training callbacks"""
        callbacks = []
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ))
        
        # Reduce learning rate on plateau
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ))
        
        # Model checkpoint
        if save_best_model:
            callbacks.append(ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ))
        
        return callbacks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "input_shape": self.input_shape,
            "class_names": self.class_names,
            "total_parameters": self.model.count_params(),
            "trainable_parameters": sum([
                tf.keras.backend.count_params(w) for w in self.model.trainable_weights
            ]),
            "layers": len(self.model.layers),
            "is_compiled": self.model.compiled_loss is not None
        }