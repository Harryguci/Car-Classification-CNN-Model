from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


class Predictor:
    def __init__(self, model_path: str, class_names: List[str] | None = None) -> None:
        self.model_path = model_path
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names or ["Bus", "Car", "Truck"]

    @staticmethod
    def preprocess_image(img_path: str, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, img_path: str, show: bool = True) -> tuple[str, float]:
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array)
        predicted_class_index = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))

        if show:
            img = image.load_img(img_path)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Predicted: {self.class_names[predicted_class_index]} (Confidence: {confidence*100:.2f}%)")
            plt.show()

        return self.class_names[predicted_class_index], confidence


