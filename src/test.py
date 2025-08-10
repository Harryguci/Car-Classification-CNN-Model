import argparse
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from predictor import Predictor
from config import TrainingConfig


def predict_image(model_path: str, image_path: str, show_plot: bool = True) -> tuple[str, float]:
    """
    Load model and predict the class of an image.
    
    Args:
        model_path: Path to the trained model file
        image_path: Path to the image to predict
        show_plot: Whether to display the image with prediction
    
    Returns:
        Tuple of (predicted_class, confidence)
    """
    try:
        # Initialize predictor with the model
        predictor = Predictor(model_path)
        
        # Make prediction
        predicted_class, confidence = predictor.predict(image_path, show=show_plot)
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
        return predicted_class, confidence
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, 0.0
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0


def main():
    parser = argparse.ArgumentParser(description="Predict vehicle class from image")
    parser.add_argument("image_path", help="Path to the image file to predict")
    parser.add_argument("--model", "-m", default="./vehicle_best_model.keras", 
                       help="Path to the trained model (default: ./vehicle_best_model.keras)")
    parser.add_argument("--no-plot", action="store_true", 
                       help="Don't show the image plot")
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
    
    # Make prediction
    predicted_class, confidence = predict_image(
        model_path=args.model,
        image_path=args.image_path,
        show_plot=not args.no_plot
    )
    
    if predicted_class is not None:
        print(f"\nFinal Result:")
        print(f"Image: {args.image_path}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
