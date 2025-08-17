#!/usr/bin/env python3
"""
Standalone Prediction Script for Soup & Cheerios Object Detection
Usage: python predict.py --image path/to/image.jpg
"""

import argparse
import pickle
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import time
from pathlib import Path
import sys

class ObjectDetectionPredictor:
    def __init__(self, model_path="exports/model_pipeline.pkl"):
        """Initialize the predictor with a trained model"""
        self.model_path = model_path
        self.model_pipeline = None
        self.classes = ['soup', 'cheerios']
        self.class_colors = {
            'soup': (255, 102, 102),     # Red
            'cheerios': (102, 178, 255)  # Blue
        }
        self.load_model()
    
    def load_model(self):
        """Load the trained model pipeline"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_pipeline = pickle.load(f)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            print("💡 Please train the model first by running: python train_model.py")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            sys.exit(1)
    
    def predict_single(self, image_path, conf_threshold=0.5, save_annotated=False, output_dir="results"):
        """Make predictions on a single image"""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        print(f"🔍 Analyzing: {image_path}")
        
        # Make prediction
        start_time = time.time()
        predictions = self.model_pipeline.predict(image_path, conf_threshold=conf_threshold)
        processing_time = time.time() - start_time
        
        # Print results
        if predictions:
            print(f"✅ Found {len(predictions)} object(s) in {processing_time:.3f}s")
            for i, pred in enumerate(predictions):
                print(f"   Detection {i+1}: {pred['class']} (confidence: {pred['confidence']:.3f})")
        else:
            print(f"⚠️  No objects detected (confidence > {conf_threshold})")
        
        # Save annotated image if requested
        if save_annotated and predictions:
            annotated_path = self.save_annotated_image(image_path, predictions, output_dir)
            print(f"💾 Annotated image saved: {annotated_path}")
        
        return {
            'image_path': image_path,
            'predictions': predictions,
            'processing_time': processing_time,
            'num_detections': len(predictions) if predictions else 0
        }
    
    def predict_batch(self, input_folder, conf_threshold=0.5, save_annotated=False, output_dir="results"):
        """Make predictions on all images in a folder"""
        input_path = Path(input_folder)
        
        if not input_path.exists():
            print(f"❌ Input folder not found: {input_folder}")
            return None
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"❌ No image files found in: {input_folder}")
            return None
        
        print(f"📁 Processing {len(image_files)} images from: {input_folder}")
        
        results = []
        total_detections = 0
        total_processing_time = 0
        
        for image_file in image_files:
            result = self.predict_single(
                str(image_file), 
                conf_threshold=conf_threshold,
                save_annotated=save_annotated,
                output_dir=output_dir
            )
            
            if result:
                results.append(result)
                total_detections += result['num_detections']
                total_processing_time += result['processing_time']
        
        # Print batch summary
        print(f"\n📊 Batch Processing Summary:")
        print(f"   Total Images: {len(image_files)}")
        print(f"   Total Detections: {total_detections}")
        print(f"   Average Processing Time: {total_processing_time/len(image_files):.3f}s")
        print(f"   Total Processing Time: {total_processing_time:.3f}s")
        
        # Save batch results
        if results:
            self.save_batch_results(results, output_dir)
        
        return results
    
    def save_annotated_image(self, image_path, predictions, output_dir="results"):
        """Save image with detection annotations"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        
        # Draw predictions
        for pred in predictions:
            # Get bounding box coordinates
            x1, y1, x2, y2 = [int(coord) for coord in pred['bbox_xyxy']]
            
            # Get class info
            class_name = pred['class']
            confidence = pred['confidence']
            color = self.class_colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Save annotated image
        input_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{input_name}_detected.jpg")
        cv2.imwrite(output_path, image)
        
        return output_path
    
    def save_batch_results(self, results, output_dir="results"):
        """Save batch processing results as JSON"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = []
        for result in results:
            json_result = {
                'image_path': result['image_path'],
                'processing_time': result['processing_time'],
                'num_detections': result['num_detections'],
                'predictions': []
            }
            
            for pred in result.get('predictions', []):
                json_pred = {
                    'class': pred['class'],
                    'confidence': float(pred['confidence']),
                    'bbox_xywh': [float(x) for x in pred['bbox']],
                    'bbox_xyxy': [float(x) for x in pred['bbox_xyxy']]
                }
                json_result['predictions'].append(json_pred)
            
            json_results.append(json_result)
        
        # Save JSON file
        output_path = os.path.join(output_dir, "batch_results.json")
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Batch results saved: {output_path}")
        return output_path
    
    def predict_with_visualization(self, image_path, conf_threshold=0.5, show_image=False):
        """Make prediction and optionally display the result"""
        result = self.predict_single(image_path, conf_threshold=conf_threshold, save_annotated=True)
        
        if show_image and result and result['predictions']:
            try:
                import matplotlib.pyplot as plt
                
                # Load original and annotated images
                original = cv2.imread(image_path)
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                
                annotated_path = self.save_annotated_image(image_path, result['predictions'])
                annotated = cv2.imread(annotated_path)
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Create subplot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                
                ax1.imshow(original)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                ax2.imshow(annotated)
                ax2.set_title(f"Detections ({len(result['predictions'])} objects)")
                ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("💡 Install matplotlib to display images: pip install matplotlib")
        
        return result

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Soup & Cheerios Object Detection Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --image test.jpg
  python predict.py --image test.jpg --conf 0.7 --save
  python predict.py --batch --input_folder images/ --output_folder results/
  python predict.py --image test.jpg --show
        """
    )
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    parser.add_argument('--input_folder', type=str, help='Input folder for batch processing')
    
    # Output options
    parser.add_argument('--output_folder', type=str, default='results', help='Output folder for results')
    parser.add_argument('--save', action='store_true', help='Save annotated images')
    parser.add_argument('--show', action='store_true', help='Display results (requires matplotlib)')
    
    # Model options
    parser.add_argument('--model', type=str, default='exports/model_pipeline.pkl', help='Path to model file')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0.1-1.0)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        parser.error("Must specify either --image or --batch")
    
    if args.batch and not args.input_folder:
        parser.error("--batch requires --input_folder")
    
    if args.conf < 0.1 or args.conf > 1.0:
        parser.error("Confidence threshold must be between 0.1 and 1.0")
    
    # Initialize predictor
    print("🚀 Initializing Object Detection Predictor...")
    predictor = ObjectDetectionPredictor(model_path=args.model)
    
    # Run prediction
    try:
        if args.batch:
            # Batch processing
            results = predictor.predict_batch(
                input_folder=args.input_folder,
                conf_threshold=args.conf,
                save_annotated=args.save,
                output_dir=args.output_folder
            )
        else:
            # Single image processing
            if args.show:
                result = predictor.predict_with_visualization(
                    image_path=args.image,
                    conf_threshold=args.conf,
                    show_image=True
                )
            else:
                result = predictor.predict_single(
                    image_path=args.image,
                    conf_threshold=args.conf,
                    save_annotated=args.save,
                    output_dir=args.output_folder
                )
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        sys.exit(1)
    
    print("🎉 Prediction completed successfully!")

if __name__ == "__main__":
    main()