#!/usr/bin/env python3
"""
Streamlit Web Application for Soup & Cheerios Object Detection
"""

import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import time
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="🥣 Soup & Cheerios Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitObjectDetector:
    def __init__(self):
        self.model_pipeline = None
        self.confidence_threshold = 0.5
        self.classes = ['soup', 'cheerios']
        self.class_colors = {
            'soup': (255, 102, 102),     # Red
            'cheerios': (102, 178, 255)  # Blue
        }
        
    @st.cache_resource
    def load_model(_self, model_path="exports/model_pipeline.pkl"):
        """Load the trained model pipeline"""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"❌ Model file not found: {model_path}")
            st.info("💡 Please train the model first by running: `python train_model.py`")
            return None
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            return None
    
    def draw_predictions(self, image, predictions):
        """Draw bounding boxes and labels on image"""
        if not predictions:
            return image
            
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        
        for pred in predictions:
            # Get bounding box coordinates (xyxy format)
            x1, y1, x2, y2 = pred['bbox_xyxy']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class info
            class_name = pred['class']
            confidence = pred['confidence']
            color = self.class_colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{class_name}: {confidence:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            # Draw label background
            cv2.rectangle(
                img_cv,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_cv,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        # Convert back to PIL
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    def create_detection_summary(self, predictions):
        """Create a summary of detections"""
        if not predictions:
            return None
            
        summary_data = []
        for i, pred in enumerate(predictions):
            summary_data.append({
                'Detection #': i + 1,
                'Class': pred['class'].title(),
                'Confidence': f"{pred['confidence']:.3f}",
                'Confidence %': f"{pred['confidence']*100:.1f}%",
                'Bounding Box': f"({pred['bbox_xyxy'][0]:.0f}, {pred['bbox_xyxy'][1]:.0f}, {pred['bbox_xyxy'][2]:.0f}, {pred['bbox_xyxy'][3]:.0f})"
            })
        
        return pd.DataFrame(summary_data)
    
    def create_confidence_chart(self, predictions):
        """Create a confidence score chart"""
        if not predictions:
            return None
            
        classes = [pred['class'] for pred in predictions]
        confidences = [pred['confidence'] for pred in predictions]
        colors = [self.class_colors[cls] for cls in classes]
        
        # Convert RGB to hex for plotly
        hex_colors = [f"rgb({c[0]},{c[1]},{c[2]})" for c in colors]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"{cls.title()} #{i+1}" for i, cls in enumerate(classes)],
                y=confidences,
                marker_color=hex_colors,
                text=[f"{conf:.3f}" for conf in confidences],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Detection Confidence Scores",
            xaxis_title="Detections",
            yaxis_title="Confidence Score",
            showlegend=False,
            height=400
        )
        
        return fig

def main():
    # Initialize detector
    detector = StreamlitObjectDetector()
    
    # Header
    st.title("🥣 Soup & Cheerios Object Detection")
    st.markdown("""
    **Upload an image to detect Soup and Cheerios objects with confidence scores!**
    
    This AI model was trained on synthetic data to identify real-world objects.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        detector.confidence_threshold = confidence_threshold
        
        st.markdown("---")
        
        # Model info
        st.header("📊 Model Info")
        st.info("""
        **Model**: YOLOv8n
        **Classes**: Soup, Cheerios
        **Training**: Synthetic Data
        **Framework**: Ultralytics
        """)
        
        # Load model
        st.header("🔄 Model Status")
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
        
        detector.model_pipeline = detector.load_model()
        
        if detector.model_pipeline:
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Model not loaded")
            st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing soup or cheerios"
        )
        
        # Example images
        st.markdown("### 🖼️ Or try example images:")
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("🍲 Soup Example"):
                st.info("Load a soup example image here")
        
        with example_col2:
            if st.button("🥣 Cheerios Example"):
                st.info("Load a cheerios example image here")
        
        with example_col3:
            if st.button("🍽️ Mixed Example"):
                st.info("Load a mixed example image here")
    
    with col2:
        st.header("🎯 Detection Results")
        
        if uploaded_file is not None:
            # Load and display original image
            image = Image.open(uploaded_file)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📸 Original", "🎯 Detections", "📊 Analysis"])
            
            with tab1:
                st.subheader("Original Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.markdown(f"""
                **Image Info:**
                - Size: {image.size[0]} × {image.size[1]} pixels
                - Format: {image.format}
                - Mode: {image.mode}
                """)
            
            with tab2:
                st.subheader("Detection Results")
                
                # Run prediction
                with st.spinner("🔍 Analyzing image..."):
                    start_time = time.time()
                    
                    # Save uploaded file temporarily
                    temp_path = f"temp_upload.{uploaded_file.name.split('.')[-1]}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Make predictions
                        predictions = detector.model_pipeline.predict(
                            temp_path, 
                            conf_threshold=confidence_threshold
                        )
                        
                        processing_time = time.time() - start_time
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
                        predictions = []
                        processing_time = 0
                
                # Display results
                if predictions:
                    st.success(f"✅ Found {len(predictions)} object(s) in {processing_time:.2f}s")
                    
                    # Draw predictions on image
                    annotated_image = detector.draw_predictions(image, predictions)
                    st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                    
                    # Download button for annotated image
                    buf = io.BytesIO()
                    annotated_image.save(buf, format='PNG')
                    
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buf.getvalue(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/png"
                    )
                    
                else:
                    st.warning(f"⚠️ No objects detected (confidence > {confidence_threshold})")
                    st.image(image, caption="No Detections", use_column_width=True)
            
            with tab3:
                st.subheader("Detection Analysis")
                
                if predictions:
                    # Summary table
                    summary_df = detector.create_detection_summary(predictions)
                    if summary_df is not None:
                        st.markdown("#### 📋 Detection Summary")
                        st.dataframe(summary_df, use_container_width=True)
                    
                    # Confidence chart
                    fig = detector.create_confidence_chart(predictions)
                    if fig is not None:
                        st.markdown("#### 📈 Confidence Scores")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Class distribution
                    class_counts = {}
                    for pred in predictions:
                        class_name = pred['class']
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    if class_counts:
                        st.markdown("#### 🍯 Class Distribution")
                        
                        # Create pie chart
                        fig_pie = px.pie(
                            values=list(class_counts.values()),
                            names=[name.title() for name in class_counts.keys()],
                            title="Distribution of Detected Objects"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Performance metrics
                    st.markdown("#### ⚡ Performance Metrics")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Objects Found", len(predictions))
                    
                    with metric_col2:
                        avg_confidence = np.mean([p['confidence'] for p in predictions])
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    with metric_col3:
                        max_confidence = max([p['confidence'] for p in predictions])
                        st.metric("Max Confidence", f"{max_confidence:.3f}")
                    
                    with metric_col4:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                
                else:
                    st.info("📈 Upload an image and run detection to see analysis")
        
        else:
            st.info("👆 Please upload an image to start detection")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🚀 Built with Streamlit and YOLOv8 | 🎯 Soup & Cheerios Detection System</p>
        <p>💡 Model trained on synthetic data for real-world performance</p>
    </div>
    """, unsafe_allow_html=True)

# Additional utility functions
def create_demo_mode():
    """Create a demo mode with sample images"""
    st.sidebar.markdown("---")
    st.sidebar.header("🎮 Demo Mode")
    
    if st.sidebar.button("🎪 Run Demo"):
        st.balloons()
        st.success("🎉 Demo mode activated! Try uploading sample images.")

def show_model_architecture():
    """Display model architecture information"""
    with st.expander("🏗️ Model Architecture Details"):
        st.markdown("""
        **YOLOv8 Nano Architecture:**
        - **Backbone**: CSPDarknet53
        - **Neck**: PANet
        - **Head**: Detection Head
        - **Parameters**: ~3.2M
        - **Input Size**: 640×640
        - **Output**: Bounding boxes + class probabilities
        
        **Training Details:**
        - **Synthetic Data**: High-fidelity digital twin simulation
        - **Classes**: 2 (Soup, Cheerios)
        - **Augmentation**: Mosaic, Mixup, HSV, Geometric
        - **Optimizer**: SGD with momentum
        """)

if __name__ == "__main__":
    # Add demo mode and architecture info
    create_demo_mode()
    show_model_architecture()
    
    # Run main app
    main()