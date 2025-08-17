# 🥣 Multi-Class Object Detection: Soup & Cheerios

A robust object detection system built with YOLOv8 to identify **Soup** and **Cheerios** in real-world images using synthetic training data from FalconCloud/Duality's digital twin simulation platform.

## 🎯 Project Overview

This project demonstrates state-of-the-art object detection capabilities by training a model on synthetic data that generalizes well to real-world test images. The system can simultaneously detect and classify two distinct object classes with high accuracy.

### 🏆 Key Features

- **Multi-Class Detection**: Detects both Soup (class 0) and Cheerios (class 1)
- **Synthetic Data Training**: Leverages high-fidelity synthetic datasets
- **Real-World Testing**: Validates performance on actual photographs
- **Web Interface**: Interactive Streamlit app for easy predictions
- **Model Export**: Complete pipeline saved as pickle for deployment
- **Comprehensive Metrics**: Detailed evaluation and performance reports

## 📁 Project Structure

```
soup-cheerios-detection/
├── 📄 README.md                    # This documentation
├── 📄 requirements.txt             # Python dependencies
├── 🐍 train_model.py              # Main training script
├── 🐍 streamlit_app.py            # Web interface
├── 🐍 predict.py                  # Standalone prediction script
├── 🐍 utils.py                    # Helper functions
├── 📄 config.yaml                 # Configuration settings
├── 📁 models/                     # Trained model weights
├── 📁 exports/                    # Exported pickle files
│   ├── model_pipeline.pkl         # Complete prediction pipeline
│   └── model_weights.pkl          # Model weights only
├── 📁 results/                    # Training results and metrics
├── 📁 configs/                    # Dataset configurations
└── 📁 utils/                      # Utility scripts
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/soup-cheerios-detection.git
cd soup-cheerios-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
Ensure your dataset follows this structure:
```
/kaggle/input/multi-class-object-detection-challenge/
├── Starter_Dataset/
│   ├── train/
│   │   ├── images/           # Training images (.png)
│   │   └── labels/           # Training labels (.txt)
│   └── val/
│       ├── images/           # Validation images (.png)
│       └── labels/           # Validation labels (.txt)
└── testImages/
    └── images/               # Test images (.png)
```

### 4. Train Model
```bash
python train_model.py
```

### 5. Run Web Interface
```bash
streamlit run streamlit_app.py
```

### 6. Make Predictions
```bash
python predict.py --image path/to/your/image.jpg
```

## 🔧 Configuration

### Dataset Classes
- **Class 0**: Soup 🍲
- **Class 1**: Cheerios 🥣

### Model Architecture
- **Base Model**: YOLOv8n (Nano - optimized for speed)
- **Input Size**: 640x640 pixels
- **Framework**: Ultralytics YOLOv8
- **Export Format**: PyTorch (.pt) + Pickle (.pkl)

### Training Parameters
```python
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
PATIENCE = 20  # Early stopping
```

## 📊 Model Performance

### Metrics
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision from IoU=0.5 to 0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)

### Expected Performance
| Metric | Soup | Cheerios | Overall |
|--------|------|----------|---------|
| Precision | 0.85+ | 0.82+ | 0.83+ |
| Recall | 0.80+ | 0.78+ | 0.79+ |
| mAP50 | 0.88+ | 0.85+ | 0.86+ |

## 🎮 Usage Examples

### Training a New Model
```python
from train_model import ObjectDetectionTrainer

trainer = ObjectDetectionTrainer()
trainer.setup_directory_structure()
trainer.create_dataset_yaml()
trainer.train_model(epochs=100, batch_size=16)
trainer.export_model_pickle()
```

### Making Predictions
```python
import pickle

# Load the trained pipeline
with open('exports/model_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Make predictions
predictions = pipeline.predict('path/to/image.jpg', conf_threshold=0.5)

for pred in predictions:
    print(f"Detected: {pred['class']} (confidence: {pred['confidence']:.2f})")
```

### Using the Web Interface
1. Launch Streamlit app: `streamlit run streamlit_app.py`
2. Upload an image using the file uploader
3. Adjust confidence threshold slider
4. View real-time detection results
5. Download annotated images

## 🧪 Testing & Validation

### Dataset Validation
The system automatically validates:
- Image-label pair consistency
- Annotation format correctness
- Class distribution balance
- File path accessibility

### Model Testing
```bash
# Test on validation set
python -c "from train_model import ObjectDetectionTrainer; trainer = ObjectDetectionTrainer(); trainer.evaluate_model()"

# Test on custom images
python predict.py --batch --input_folder test_images/ --output_folder results/
```

## 📈 Data Augmentation

The training pipeline includes extensive augmentation:
- **Geometric**: Rotation, scaling, translation, flipping
- **Color**: HSV adjustments, brightness, contrast
- **Advanced**: Mixup, mosaic, copy-paste augmentation
- **Noise**: Gaussian noise, blur effects

## 🔄 Model Export Options

### 1. Complete Pipeline (Recommended)
```python
# Exports model + preprocessing + postprocessing
pipeline = PredictionPipeline(model, classes)
pickle.dump(pipeline, open('model_pipeline.pkl', 'wb'))
```

### 2. Model Weights Only
```python
# Exports just the trained weights
pickle.dump(model, open('model_weights.pkl', 'wb'))
```

### 3. ONNX Format
```python
# For deployment optimization
model.export(format='onnx', imgsz=640)
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA/GPU Not Detected**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. Dataset Path Issues**
- Verify dataset structure matches expected format
- Check file permissions and accessibility
- Ensure images and labels have matching filenames

**3. Memory Issues**
- Reduce batch size: `batch_size=8` or `batch_size=4`
- Use smaller image size: `img_size=416`
- Close other applications

**4. Poor Performance**
- Increase training epochs
- Check annotation quality
- Add more diverse training data
- Adjust augmentation parameters

### Error Solutions

| Error | Solution |
|-------|----------|
| `FileNotFoundError` | Check dataset paths in config |
| `CUDA out of memory` | Reduce batch size |
| `Low mAP scores` | Increase epochs or data quality |
| `Import errors` | Reinstall requirements |

## 🌟 Advanced Features

### Custom Training
```python
# Advanced training configuration
trainer.train_model(
    epochs=200,
    img_size=832,
    batch_size=8,
    lr0=0.01,
    weight_decay=0.0005,
    warmup_epochs=3
)
```

### Hyperparameter Tuning
```bash
# Automatic hyperparameter optimization
yolo tune model=yolov8n.pt data=configs/dataset.yaml epochs=30 iterations=100
```

### Model Ensemble
```python
# Combine multiple models for better accuracy
from utils import ModelEnsemble

ensemble = ModelEnsemble([
    'models/yolov8n_best.pt',
    'models/yolov8s_best.pt'
])
predictions = ensemble.predict('image.jpg')
```

## 📚 API Reference

### PredictionPipeline Class
```python
class PredictionPipeline:
    def predict(image_path, conf_threshold=0.5)
        """Single image prediction"""
        
    def predict_batch(image_paths, conf_threshold=0.5)
        """Batch prediction"""
        
    def visualize_predictions(image_path, save_path=None)
        """Create annotated image"""
```

### ObjectDetectionTrainer Class
```python
class ObjectDetectionTrainer:
    def setup_directory_structure()
        """Initialize project directories"""
        
    def train_model(epochs, batch_size, img_size)
        """Train YOLO model"""
        
    def export_model_pickle()
        """Export trained model as pickle"""
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -m 'Add feature'`
4. **Push** to branch: `git push origin feature-name`
5. **Submit** a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pre-commit black flake8 pytest

# Set up pre-commit hooks
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8**: State-of-the-art object detection framework
- **FalconCloud/Duality**: High-fidelity synthetic data generation
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library

## 📧 Contact

- **GitHub**: [yourusername](https://github.com/yourusername)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🏗️ Roadmap

### Version 2.0 Features
- [ ] **Real-time video detection**
- [ ] **Mobile app deployment**
- [ ] **Cloud API integration**
- [ ] **Additional object classes**
- [ ] **Performance optimizations**
- [ ] **Docker containerization**

### Future Enhancements
- [ ] **3D object detection**
- [ ] **Segmentation masks**
- [ ] **Multi-camera fusion**
- [ ] **Edge device deployment**

---

## 🚀 Quick Commands Cheat Sheet

```bash
# Setup
git clone <repo> && cd soup-cheerios-detection
pip install -r requirements.txt

# Training
python train_model.py

# Web Interface
streamlit run streamlit_app.py

# Prediction
python predict.py --image test.jpg

# Batch Processing
python predict.py --batch --input_folder images/

# Export Model
python -c "from train_model import ObjectDetectionTrainer; ObjectDetectionTrainer().export_model_pickle()"
```

**Happy Detecting! 🎯**