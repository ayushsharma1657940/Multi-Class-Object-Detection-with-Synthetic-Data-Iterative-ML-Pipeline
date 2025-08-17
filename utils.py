#!/usr/bin/env python3
"""
Utility Functions for Soup & Cheerios Object Detection Project
Contains helper functions for data processing, visualization, evaluation, and model management.
"""

import os
import json
import yaml
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time
import shutil
from typing import List, Dict, Tuple, Optional, Union
import logging
from datetime import datetime
import hashlib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetUtils:
    """Utilities for dataset management and validation"""
    
    @staticmethod
    def validate_yolo_format(label_path: str) -> bool:
        """Validate YOLO format annotation file"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    return False
                
                # Check class_id is integer
                class_id = int(parts[0])
                
                # Check coordinates are floats between 0 and 1
                coords = [float(x) for x in parts[1:]]
                if any(coord < 0 or coord > 1 for coord in coords):
                    return False
                    
            return True
        except Exception:
            return False
    
    @staticmethod
    def convert_bbox_format(bbox: List[float], from_format: str, to_format: str, img_width: int, img_height: int) -> List[float]:
        """Convert bounding box between different formats"""
        if from_format == "yolo" and to_format == "xyxy":
            # YOLO (x_center, y_center, width, height) normalized to XYXY (x1, y1, x2, y2) absolute
            x_center, y_center, width, height = bbox
            x1 = (x_center - width/2) * img_width
            y1 = (y_center - height/2) * img_height
            x2 = (x_center + width/2) * img_width
            y2 = (y_center + height/2) * img_height
            return [x1, y1, x2, y2]
        
        elif from_format == "xyxy" and to_format == "yolo":
            # XYXY absolute to YOLO normalized
            x1, y1, x2, y2 = bbox
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            return [x_center, y_center, width, height]
        
        elif from_format == "xywh" and to_format == "xyxy":
            # XYWH (x, y, width, height) to XYXY
            x, y, width, height = bbox
            return [x, y, x + width, y + height]
        
        elif from_format == "xyxy" and to_format == "xywh":
            # XYXY to XYWH
            x1, y1, x2, y2 = bbox
            return [x1, y1, x2 - x1, y2 - y1]
        
        else:
            raise ValueError(f"Conversion from {from_format} to {to_format} not implemented")
    
    @staticmethod
    def analyze_dataset(dataset_path: str) -> Dict:
        """Analyze dataset and return statistics"""
        dataset_path = Path(dataset_path)
        
        stats = {
            'train_images': 0,
            'train_labels': 0,
            'val_images': 0,
            'val_labels': 0,
            'test_images': 0,
            'class_distribution': {'soup': 0, 'cheerios': 0},
            'image_sizes': [],
            'annotation_counts': []
        }
        
        # Count files
        train_img_path = dataset_path / "train" / "images"
        train_lbl_path = dataset_path / "train" / "labels"
        val_img_path = dataset_path / "val" / "images"
        val_lbl_path = dataset_path / "val" / "labels"
        test_img_path = dataset_path / "testImages" / "images"
        
        if train_img_path.exists():
            stats['train_images'] = len(list(train_img_path.glob("*.png")))
        if train_lbl_path.exists():
            stats['train_labels'] = len(list(train_lbl_path.glob("*.txt")))
        if val_img_path.exists():
            stats['val_images'] = len(list(val_img_path.glob("*.png")))
        if val_lbl_path.exists():
            stats['val_labels'] = len(list(val_lbl_path.glob("*.txt")))
        if test_img_path.exists():
            stats['test_images'] = len(list(test_img_path.glob("*.png")))
        
        # Analyze annotations
        for label_file in train_lbl_path.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                stats['annotation_counts'].append(len(lines))
                
                for line in lines:
                    class_id = int(line.strip().split()[0])
                    if class_id == 0:
                        stats['class_distribution']['soup'] += 1
                    elif class_id == 1:
                        stats['class_distribution']['cheerios'] += 1
        
        return stats

class VisualizationUtils:
    """Utilities for creating visualizations and plots"""
    
    @staticmethod
    def plot_training_metrics(results_path: str, save_path: Optional[str] = None):
        """Plot training metrics from YOLOv8 results"""
        try:
            results_csv = Path(results_path) / "results.csv"
            if not results_csv.exists():
                logger.warning(f"Results file not found: {results_csv}")
                return
            
            # Read results
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()  # Remove whitespace
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Training Metrics', fontsize=16)
            
            # Loss plots
            if 'train/box_loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
                axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
                axes[0, 0].set_title('Box Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # mAP plots
            if 'metrics/mAP50(B)' in df.columns:
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
                axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
                axes[0, 1].set_title('Mean Average Precision')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('mAP')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Precision/Recall
            if 'metrics/precision(B)' in df.columns:
                axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
                axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
                axes[1, 0].set_title('Precision & Recall')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            if 'lr/pg0' in df.columns:
                axes[1, 1].plot(df['epoch'], df['lr/pg0'])
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training metrics plot saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training metrics: {str(e)}")
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_detection_samples(image_paths: List[str], predictions_list: List[List[Dict]], 
                             save_path: Optional[str] = None, max_samples: int = 6):
        """Plot sample detection results"""
        n_samples = min(len(image_paths), max_samples)
        cols = 3
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        class_colors = {'soup': 'red', 'cheerios': 'blue'}
        
        for i in range(n_samples):
            row = i // cols
            col = i % cols
            
            # Load and display image
            image = cv2.imread(image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(image)
            axes[row, col].set_title(f"Sample {i+1}")
            axes[row, col].axis('off')
            
            # Draw predictions
            predictions = predictions_list[i]
            for pred in predictions:
                x1, y1, x2, y2 = pred['bbox_xyxy']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color=class_colors.get(pred['class'], 'green'), 
                                   linewidth=2)
                axes[row, col].add_patch(rect)
                
                # Add label
                axes[row, col].text(x1, y1-5, f"{pred['class']}: {pred['confidence']:.2f}",
                                  color=class_colors.get(pred['class'], 'green'), 
                                  fontsize=8, weight='bold')
        
        # Hide empty subplots
        for i in range(n_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detection samples plot saved: {save_path}")
        
        plt.show()

class ModelUtils:
    """Utilities for model management and optimization"""
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict:
        """Get information about a trained model"""
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
                # Try to extract model info
                if hasattr(model, 'model'):
                    yolo_model = model.model
                else:
                    yolo_model = model
                
                info = {
                    'model_type': 'YOLOv8',
                    'classes': getattr(model, 'classes', ['soup', 'cheerios']),
                    'num_classes': len(getattr(model, 'classes', ['soup', 'cheerios'])),
                    'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                    'framework': 'Ultralytics'
                }
                
            elif model_path.endswith('.pt'):
                # PyTorch model
                checkpoint = torch.load(model_path, map_location='cpu')
                info = {
                    'model_type': 'YOLOv8',
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'best_fitness': checkpoint.get('best_fitness', 'Unknown'),
                    'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                    'framework': 'Ultralytics'
                }
            
            else:
                info = {'error': 'Unsupported model format'}
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def optimize_model_for_inference(model_path: str, output_path: str, format: str = 'onnx'):
        """Optimize model for inference"""
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Export to specified format
            model.export(format=format, imgsz=640)
            
            logger.info(f"Model exported to {format} format")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            return False
    
    @staticmethod
    def compress_model(model_path: str, output_path: str, compression_level: int = 9):
        """Compress model file using gzip"""
        import gzip
        
        try:
            with open(model_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            original_size = os.path.getsize(model_path)
            compressed_size = os.path.getsize(output_path)
            compression_ratio = compressed_size / original_size
            
            logger.info(f"Model compressed: {original_size/1024/1024:.2f}MB -> {compressed_size/1024/1024:.2f}MB (ratio: {compression_ratio:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Error compressing model: {str(e)}")
            return False

class EvaluationUtils:
    """Utilities for model evaluation and metrics"""
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_map(predictions: List[Dict], ground_truths: List[Dict], iou_threshold: float = 0.5) -> Dict:
        """Calculate mean Average Precision (mAP)"""
        # This is a simplified implementation
        # For full mAP calculation, use pycocotools or ultralytics evaluation
        
        classes = ['soup', 'cheerios']
        ap_scores = {}
        
        for class_name in classes:
            # Filter predictions and ground truths for this class
            class_predictions = [p for p in predictions if p['class'] == class_name]
            class_ground_truths = [gt for gt in ground_truths if gt['class'] == class_name]
            
            if not class_ground_truths:
                ap_scores[class_name] = 0.0
                continue
            
            # Sort predictions by confidence
            class_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision and recall
            tp = 0
            fp = 0
            
            for pred in class_predictions:
                best_iou = 0
                best_gt_idx = -1
                
                for i, gt in enumerate(class_ground_truths):
                    iou = EvaluationUtils.calculate_iou(pred['bbox_xyxy'], gt['bbox_xyxy'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i
                
                if best_iou >= iou_threshold:
                    tp += 1
                    # Remove matched ground truth to avoid double matching
                    class_ground_truths.pop(best_gt_idx)
                else:
                    fp += 1
            
            # Calculate AP (simplified)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(class_ground_truths) if class_ground_truths else 0
            
            ap_scores[class_name] = precision  # Simplified AP calculation
        
        # Calculate mAP
        map_score = np.mean(list(ap_scores.values()))
        
        return {
            'mAP': map_score,
            'AP_per_class': ap_scores
        }

class FileUtils:
    """Utilities for file and directory operations"""
    
    @staticmethod
    def create_project_structure(base_path: str):
        """Create standard project directory structure"""
        directories = [
            'models',
            'results',
            'exports',
            'configs',
            'utils',
            'data/train/images',
            'data/train/labels',
            'data/val/images',
            'data/val/labels',
            'data/test/images'
        ]
        
        base_path = Path(base_path)
        for directory in directories:
            (base_path / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project structure created at: {base_path}")
    
    @staticmethod
    def backup_files(source_paths: List[str], backup_dir: str):
        """Create backup copies of important files"""
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for source_path in source_paths:
            source = Path(source_path)
            if source.exists():
                backup_name = f"{source.stem}_{timestamp}{source.suffix}"
                backup_path = backup_dir / backup_name
                shutil.copy2(source, backup_path)
                logger.info(f"Backed up: {source} -> {backup_path}")
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Save configuration as YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved: {config_path}")
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded: {config_path}")
        return config

class Logger:
    """Custom logger for the project"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def warning(self, message: str):
        self.logger.warning(message)

# Example usage and testing functions
def test_utilities():
    """Test various utility functions"""
    print("🧪 Testing utility functions...")
    
    # Test bbox conversion
    yolo_bbox = [0.5, 0.5, 0.4, 0.3]  # center_x, center_y, width, height
    xyxy_bbox = DatasetUtils.convert_bbox_format(yolo_bbox, "yolo", "xyxy", 640, 480)
    print(f"YOLO {yolo_bbox} -> XYXY {xyxy_bbox}")
    
    # Test IoU calculation
    box1 = [100, 100, 200, 200]
    box2 = [150, 150, 250, 250]
    iou = EvaluationUtils.calculate_iou(box1, box2)
    print(f"IoU between {box1} and {box2}: {iou:.3f}")
    
    print("✅ Utility tests completed!")

if __name__ == "__main__":
    test_utilities()