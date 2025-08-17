#!/usr/bin/env python3
"""
Multi-Class Object Detection Training Script
Classes: Soup (0), Cheerios (1)
"""

import os
import yaml
import pickle
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

class ObjectDetectionTrainer:
    def __init__(self, data_path="/kaggle/input/multi-class-object-detection-challenge"):
        self.data_path = Path(data_path)
        self.project_name = "soup_cheerios_detection"
        self.classes = ['soup', 'cheerios']
        self.model = None
        self.results = None
        
    def setup_directory_structure(self):
        """Create necessary directories for the project"""
        directories = [
            'models',
            'results',
            'exports',
            'configs',
            'utils'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        print("✅ Directory structure created")
        
    def create_dataset_yaml(self):
        """Create YOLO dataset configuration file"""
        dataset_config = {
            'path': str(self.data_path / "Starter_Dataset"),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'testImages/images',
            'nc': 2,  # number of classes
            'names': self.classes
        }
        
        config_path = Path('configs/dataset.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
            
        print(f"✅ Dataset config created: {config_path}")
        return config_path
        
    def validate_dataset(self):
        """Validate dataset structure and annotations"""
        train_images = list((self.data_path / "Starter_Dataset/train/images").glob("*.png"))
        train_labels = list((self.data_path / "Starter_Dataset/train/labels").glob("*.txt"))
        val_images = list((self.data_path / "Starter_Dataset/val/images").glob("*.png"))
        val_labels = list((self.data_path / "Starter_Dataset/val/labels").glob("*.txt"))
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training Images: {len(train_images)}")
        print(f"   Training Labels: {len(train_labels)}")
        print(f"   Validation Images: {len(val_images)}")
        print(f"   Validation Labels: {len(val_labels)}")
        
        # Validate some annotations
        if train_labels:
            with open(train_labels[0], 'r') as f:
                sample_annotation = f.read().strip()
                print(f"   Sample annotation: {sample_annotation}")
                
        return len(train_images) > 0 and len(val_images) > 0
        
    def train_model(self, epochs=100, img_size=640, batch_size=16):
        """Train YOLOv8 model"""
        print("🚀 Starting model training...")
        
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # nano version for faster training
        
        # Training parameters
        train_params = {
            'data': 'configs/dataset.yaml',
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': self.project_name,
            'project': 'results',
            'patience': 20,
            'save_period': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'augment': True,
            'mixup': 0.1,
            'mosaic': 1.0,
            'copy_paste': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
        }
        
        # Train the model
        self.results = self.model.train(**train_params)
        
        print("✅ Training completed!")
        return self.results
        
    def evaluate_model(self):
        """Evaluate model performance"""
        if not self.model:
            print("❌ No model found. Please train first.")
            return
            
        print("📈 Evaluating model...")
        
        # Validate on validation set
        val_results = self.model.val(data='configs/dataset.yaml')
        
        # Print metrics
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")
        
        return val_results
        
    def export_model_pickle(self, model_path=None):
        """Export trained model and preprocessing pipeline as pickle"""
        if not self.model:
            print("❌ No model to export. Please train first.")
            return
            
        print("💾 Exporting model as pickle...")
        
        # Use best model if available
        if model_path is None:
            model_path = f"results/{self.project_name}/weights/best.pt"
            
        # Load the best model
        best_model = YOLO(model_path)
        
        # Create prediction pipeline
        class PredictionPipeline:
            def __init__(self, model, classes):
                self.model = model
                self.classes = classes
                
            def predict(self, image_path, conf_threshold=0.5):
                """Make predictions on an image"""
                results = self.model(image_path, conf=conf_threshold)
                
                predictions = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i in range(len(boxes)):
                            pred = {
                                'class': self.classes[int(boxes.cls[i])],
                                'confidence': float(boxes.conf[i]),
                                'bbox': boxes.xywh[i].tolist(),  # x_center, y_center, width, height
                                'bbox_xyxy': boxes.xyxy[i].tolist()  # x1, y1, x2, y2
                            }
                            predictions.append(pred)
                            
                return predictions
                
            def predict_batch(self, image_paths, conf_threshold=0.5):
                """Make predictions on multiple images"""
                batch_predictions = {}
                for img_path in image_paths:
                    batch_predictions[img_path] = self.predict(img_path, conf_threshold)
                return batch_predictions
                
        # Create pipeline instance
        pipeline = PredictionPipeline(best_model, self.classes)
        
        # Save pipeline as pickle
        pickle_path = "exports/model_pipeline.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(pipeline, f)
            
        # Also save just the model weights
        model_pickle_path = "exports/model_weights.pkl"
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(best_model, f)
            
        print(f"✅ Model pipeline exported to: {pickle_path}")
        print(f"✅ Model weights exported to: {model_pickle_path}")
        
        return pickle_path, model_pickle_path
        
    def test_on_test_set(self):
        """Test model on test images"""
        if not self.model:
            print("❌ No model found. Please train first.")
            return
            
        test_images = list((self.data_path / "testImages/images").glob("*.png"))
        
        if not test_images:
            print("❌ No test images found.")
            return
            
        print(f"🧪 Testing on {len(test_images)} test images...")
        
        results = []
        for img_path in test_images[:5]:  # Test on first 5 images
            pred_result = self.model(str(img_path))
            results.append({
                'image': img_path.name,
                'predictions': pred_result
            })
            
        return results
        
    def create_sample_predictions(self):
        """Create sample predictions for demonstration"""
        test_images = list((self.data_path / "testImages/images").glob("*.png"))
        
        if test_images:
            # Load the exported model
            with open("exports/model_pipeline.pkl", 'rb') as f:
                pipeline = pickle.load(f)
                
            # Make predictions
            sample_image = str(test_images[0])
            predictions = pipeline.predict(sample_image)
            
            print(f"📸 Sample predictions for {test_images[0].name}:")
            for pred in predictions:
                print(f"   {pred['class']}: {pred['confidence']:.2f}")
                
            return predictions
            
    def generate_training_report(self):
        """Generate comprehensive training report"""
        if not self.results:
            print("❌ No training results found.")
            return
            
        report = {
            'model_type': 'YOLOv8n',
            'classes': self.classes,
            'training_completed': True,
            'model_files': {
                'best_weights': f"results/{self.project_name}/weights/best.pt",
                'last_weights': f"results/{self.project_name}/weights/last.pt",
                'pickle_pipeline': "exports/model_pipeline.pkl",
                'pickle_weights': "exports/model_weights.pkl"
            }
        }
        
        # Save report
        with open("results/training_report.yaml", 'w') as f:
            yaml.dump(report, f)
            
        print("✅ Training report generated")
        return report

def main():
    """Main training pipeline"""
    trainer = ObjectDetectionTrainer()
    
    # Setup
    trainer.setup_directory_structure()
    trainer.create_dataset_yaml()
    
    # Validate dataset
    if not trainer.validate_dataset():
        print("❌ Dataset validation failed!")
        return
        
    # Train model
    trainer.train_model(epochs=50, batch_size=8)  # Reduced for demo
    
    # Evaluate
    trainer.evaluate_model()
    
    # Export model
    trainer.export_model_pickle()
    
    # Test predictions
    trainer.test_on_test_set()
    trainer.create_sample_predictions()
    
    # Generate report
    trainer.generate_training_report()
    
    print("🎉 Training pipeline completed successfully!")
    print("📁 Check the following files:")
    print("   - exports/model_pipeline.pkl (for predictions)")
    print("   - exports/model_weights.pkl (model weights)")
    print("   - results/training_report.yaml (training summary)")

if __name__ == "__main__":
    main()