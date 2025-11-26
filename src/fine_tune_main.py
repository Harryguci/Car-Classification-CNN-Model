#!/usr/bin/env python3
"""
Fine-tuning script for vehicle classification model.
Uses MLModelService to fine-tune a pre-trained model with new data.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from services.ml_model_service import MLModelService

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fine_tune.log')
        ]
    )

def validate_paths(model_path: str, data_dir: str) -> bool:
    """Validate that model and data paths exist"""
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_dir):
        print(f"❌ Training data directory not found: {data_dir}")
        return False
    
    # Check if data directory has the expected structure
    expected_classes = ["car", "bus", "truck"]
    for class_name in expected_classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"❌ Expected class directory not found: {class_dir}")
            return False
        
        # Check if class directory has images
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(image_files) == 0:
            print(f"❌ No images found in class directory: {class_dir}")
            return False
        
        print(f"✅ Found {len(image_files)} images in {class_name}/")
    
    return True

def main():
    """Main fine-tuning function"""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained vehicle classification model"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to the pre-trained .keras model file"
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to training data directory (should contain car/, bus/, truck/ subdirectories)"
    )
    
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=0.0001,
        help="Learning rate for fine-tuning (default: 0.0001)"
    )
    
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=None,
        help="Number of initial layers to freeze (default: None - no freezing)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/fine_tuned_model.keras",
        help="Output path for fine-tuned model (default: models/fine_tuned_model.keras)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    print("🚗 Vehicle Classification Model Fine-tuning")
    print("=" * 50)
    
    # Validate paths
    if not validate_paths(args.model, args.data):
        sys.exit(1)
    
    try:
        # Initialize ML model service
        logger.info("Initializing ML Model Service...")
        ml_service = MLModelService(model_path=args.model)
        
        # Load the pre-trained model
        logger.info(f"Loading pre-trained model from {args.model}")
        if not ml_service.load_model(args.model):
            print("❌ Failed to load the pre-trained model")
            sys.exit(1)
        
        print("✅ Model loaded successfully")
        
        # Display model information
        model_info = ml_service.get_model_info()
        print(f"\n📊 Model Information:")
        print(f"   Input shape: {model_info['input_shape']}")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Trainable parameters: {model_info['trainable_parameters']:,}")
        print(f"   Number of layers: {model_info['layers']}")
        
        # Fine-tune the model
        print(f"\n🔄 Starting fine-tuning...")
        print(f"   Training data: {args.data}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Validation split: {args.validation_split}")
        if args.freeze_layers:
            print(f"   Freezing first {args.freeze_layers} layers")
        
        # Update model path for saving
        ml_service.model_path = args.output
        
        # Perform fine-tuning
        results = ml_service.fine_tune_model(
            training_data_dir=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split,
            freeze_layers=args.freeze_layers,
            save_best_model=True
        )
        
        # Display results
        if results["status"] == "success":
            print("\n✅ Fine-tuning completed successfully!")
            print(f"📈 Results:")
            print(f"   Training samples: {results['training_samples']}")
            print(f"   Validation samples: {results['validation_samples']}")
            print(f"   Epochs completed: {results['epochs_completed']}")
            print(f"   Final training accuracy: {results['final_accuracy']:.4f}")
            print(f"   Final validation accuracy: {results['final_val_accuracy']:.4f}")
            print(f"   Final training loss: {results['final_loss']:.4f}")
            print(f"   Final validation loss: {results['final_val_loss']:.4f}")
            print(f"   Model saved to: {results['model_path']}")
            
            # Save the final model
            if ml_service.save_model(args.output):
                print(f"💾 Final model saved to: {args.output}")
            else:
                print("⚠️  Warning: Failed to save final model")
                
        else:
            print(f"❌ Fine-tuning failed: {results['error']}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Fine-tuning interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during fine-tuning: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()