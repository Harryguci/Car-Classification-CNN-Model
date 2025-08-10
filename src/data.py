import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import TrainingConfig


class DatasetManager:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def remove_corrupt_images(self) -> None:
        for root, _, files in os.walk(self.config.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                        if img.format not in ["JPEG", "PNG", "GIF", "BMP"]:
                            raise ValueError(f"Unsupported image format: {img.format}")
                except (IOError, ValueError) as error:
                    print(f"Corrupt image removed: {file_path} - {error}")
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

    def load_dataset_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        # Set environment variable to handle non-UTF-8 filenames
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        try:
            dataset = tf.keras.utils.image_dataset_from_directory(
                self.config.data_dir,
                image_size=(self.config.img_height, self.config.img_width),
                batch_size=self.config.batch_size,
                validation_split=None,
                subset=None,
                seed=None,
                shuffle=True,
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False,
                pad_to_aspect_ratio=False,
                data_format=None,
                verbose=1,
            )
        except UnicodeDecodeError as e:
            print(f"Unicode error encountered: {e}")
            print("Attempting to load dataset with manual file handling...")
            return self._load_dataset_manually()
        
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for image_batch, label_batch in dataset:
            images.extend(image_batch.numpy())
            labels.extend(label_batch.numpy())

        X = np.array(images, dtype=np.float16) / 255.0
        y = np.array(labels, dtype=np.int32)
        return X, y

    def _load_dataset_manually(self) -> Tuple[np.ndarray, np.ndarray]:
        """Manual dataset loading to handle non-UTF-8 filenames"""
        images = []
        labels = []
        class_names = sorted([d for d in os.listdir(self.config.data_dir) 
                            if os.path.isdir(os.path.join(self.config.data_dir, d))])
        
        print(f"Found classes: {class_names}")
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.config.data_dir, class_name)
            print(f"Loading class {class_name} (index {class_idx}) from {class_dir}")
            
            try:
                files = os.listdir(class_dir)
            except UnicodeDecodeError:
                print(f"Unicode error reading directory {class_dir}, skipping...")
                continue
                
            for file_name in files:
                if not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    continue
                    
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load and resize image
                    img = tf.keras.preprocessing.image.load_img(
                        file_path, 
                        target_size=(self.config.img_height, self.config.img_width)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        if not images:
            raise ValueError("No valid images found in dataset")
            
        X = np.array(images, dtype=np.float16) / 255.0
        y = np.array(labels, dtype=np.int32)
        
        print(f"Successfully loaded {len(X)} images with shape {X.shape}")
        return X, y

    # Intentionally no TF dataset factory here; training creates datasets per fold


