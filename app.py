"""
Cloud-Optimized Flask App for CIFAR-10 Classification
Supports both Custom CNN (PyTorch) and MobileNetV2 models
"""

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import io
import base64

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch torchvision")

# TensorFlow imports
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Install with: pip install tensorflow")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
CUSTOM_CNN_PATH = 'models/best_model_cifar10.pth'
MOBILENET_PATH = 'models/cifar10_tensorflow_model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Global model variables
custom_cnn_model = None
mobilenet_model = None

# Note: The saved model is actually MobileNetV2, not a custom CNN class
# Custom CNN class definition (unused - keeping for reference)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: input 3 channels (RGB), output 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  
        
        # Second convolutional layer: input 32, output 32 feature maps
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Third convolutional layer: input 32, output 64 feature maps
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fourth convolutional layer: input 64, output 64 feature maps
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Fifth and sixth convolutional layers
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Flatten layer to convert 2D feature maps to 1D feature vector
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjusted based on your model
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        # Pass input through conv layers with batch norm and ReLU activation
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def load_custom_cnn():
    """Load the Custom CNN PyTorch model (actually MobileNetV2)"""
    global custom_cnn_model
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return None
    
    try:
        if os.path.exists(CUSTOM_CNN_PATH):
            # Load checkpoint
            checkpoint = torch.load(CUSTOM_CNN_PATH, map_location='cpu')
            
            # The saved model is actually MobileNetV2
            custom_cnn_model = models.mobilenet_v2(pretrained=False)
            custom_cnn_model.classifier[1] = torch.nn.Linear(custom_cnn_model.classifier[1].in_features, 10)
            
            # Load the state dict
            custom_cnn_model.load_state_dict(checkpoint['model_state_dict'])
            custom_cnn_model.eval()
            
            print("✅ Custom CNN model (MobileNetV2) loaded successfully!")
            return custom_cnn_model
        else:
            print(f"❌ No Custom CNN model found at {CUSTOM_CNN_PATH}")
            return None
    except Exception as e:
        print(f"❌ Error loading Custom CNN model: {e}")
        return None

def load_mobilenet():
    """Load the MobileNetV2 TensorFlow model"""
    global mobilenet_model
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available")
        return None
    
    try:
        if os.path.exists(MOBILENET_PATH):
            mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH)
            print("✅ MobileNetV2 model loaded successfully!")
            return mobilenet_model
        else:
            print(f"❌ No MobileNetV2 model found at {MOBILENET_PATH}")
            return None
    except Exception as e:
        print(f"❌ Error loading MobileNetV2 model: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_pytorch(image_file):
    """Preprocess uploaded image for PyTorch model"""
    try:
        # Define the same transforms as used in training
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        # Open and convert to RGB
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        raise Exception(f"Error preprocessing image for PyTorch: {str(e)}")

def preprocess_image_tensorflow(image_file):
    """Preprocess uploaded image for TensorFlow model"""
    try:
        # Open and convert to RGB
        image = Image.open(image_file).convert('RGB')
        image = image.resize((224, 224))  # MobileNetV2 expects 224x224
        
        # Convert to numpy array and normalize for ImageNet
        image_array = np.array(image).astype('float32') / 255.0
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        raise Exception(f"Error preprocessing image for TensorFlow: {str(e)}")

def predict_with_custom_cnn(image_tensor):
    """Make prediction using Custom CNN PyTorch model"""
    if custom_cnn_model is None:
        raise Exception("Custom CNN model not loaded")
    
    try:
        with torch.no_grad():
            outputs = custom_cnn_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Get top 5 predictions
        top_indices = torch.argsort(probabilities, descending=True)[:5]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': CIFAR10_CLASSES[idx.item()],
                'probability': float(probabilities[idx.item()]) * 100
            })
        
        return results
    except Exception as e:
        raise Exception(f"Custom CNN prediction error: {str(e)}")

def predict_with_mobilenet(image_array):
    """Make prediction using MobileNetV2 TensorFlow model"""
    if mobilenet_model is None:
        raise Exception("MobileNetV2 model not loaded")
    
    try:
        predictions = mobilenet_model.predict(image_array, verbose=0)
        probabilities = predictions[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'class': CIFAR10_CLASSES[idx],
                'probability': float(probabilities[idx]) * 100
            })
        
        return results
    except Exception as e:
        raise Exception(f"MobileNetV2 prediction error: {str(e)}")

# Load models at startup
if TORCH_AVAILABLE:
    load_custom_cnn()

if TF_AVAILABLE:
    load_mobilenet()

@app.route('/')
def index():
    """Main upload page"""
    custom_cnn_status = "✅ Loaded" if custom_cnn_model is not None else "❌ Not Loaded"
    mobilenet_status = "✅ Loaded" if mobilenet_model is not None else "❌ Not Loaded"
    
    return render_template('index.html', 
                         custom_cnn_status=custom_cnn_status,
                         mobilenet_status=mobilenet_status,
                         torch_available=TORCH_AVAILABLE,
                         tf_available=TF_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make predictions"""
    # Get selected model
    selected_model = request.form.get('model', 'custom_cnn')
    
    # Check if the selected model is available
    if selected_model == 'custom_cnn':
        if not TORCH_AVAILABLE:
            flash('PyTorch not available.')
            return redirect(url_for('index'))
        if custom_cnn_model is None:
            flash('Custom CNN model not loaded. Please check server logs.')
            return redirect(url_for('index'))
    elif selected_model == 'mobilenet':
        if not TF_AVAILABLE:
            flash('TensorFlow not available.')
            return redirect(url_for('index'))
        if mobilenet_model is None:
            flash('MobileNetV2 model not loaded. Please check server logs.')
            return redirect(url_for('index'))
    elif selected_model == 'both':
        available_models = []
        if TORCH_AVAILABLE and custom_cnn_model is not None:
            available_models.append('custom_cnn')
        if TF_AVAILABLE and mobilenet_model is not None:
            available_models.append('mobilenet')
        
        if not available_models:
            flash('No models are available for prediction.')
            return redirect(url_for('index'))
    
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            results = {}
            
            # Make predictions with selected model(s)
            if selected_model == 'custom_cnn' or selected_model == 'both':
                if TORCH_AVAILABLE and custom_cnn_model is not None:
                    file.seek(0)
                    image_tensor = preprocess_image_pytorch(file)
                    results['custom_cnn'] = predict_with_custom_cnn(image_tensor)
            
            if selected_model == 'mobilenet' or selected_model == 'both':
                if TF_AVAILABLE and mobilenet_model is not None:
                    file.seek(0)
                    image_array = preprocess_image_tensorflow(file)
                    results['mobilenet'] = predict_with_mobilenet(image_array)
            
            # Convert image to base64 for display
            file.seek(0)  # Reset file pointer
            image_b64 = base64.b64encode(file.read()).decode('utf-8')
            
            return render_template('results.html', 
                                 results=results,
                                 selected_model=selected_model,
                                 image_data=image_b64,
                                 filename=file.filename)
            
        except Exception as e:
            flash(f"Error processing image: {str(e)}")
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload an image.')
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    # Get selected model from form or query parameter
    selected_model = request.form.get('model', request.args.get('model', 'custom_cnn'))
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            results = {}
            
            # Make predictions with selected model(s)
            if selected_model == 'custom_cnn' or selected_model == 'both':
                if TORCH_AVAILABLE and custom_cnn_model is not None:
                    file.seek(0)
                    image_tensor = preprocess_image_pytorch(file)
                    results['custom_cnn'] = predict_with_custom_cnn(image_tensor)
                elif selected_model == 'custom_cnn':
                    return jsonify({'error': 'Custom CNN model not available'}), 500
            
            if selected_model == 'mobilenet' or selected_model == 'both':
                if TF_AVAILABLE and mobilenet_model is not None:
                    file.seek(0)
                    image_array = preprocess_image_tensorflow(file)
                    results['mobilenet'] = predict_with_mobilenet(image_array)
                elif selected_model == 'mobilenet':
                    return jsonify({'error': 'MobileNetV2 model not available'}), 500
            
            return jsonify({
                'success': True,
                'model': selected_model,
                'results': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    """Health check endpoint for cloud platforms"""
    return jsonify({
        'status': 'healthy',
        'pytorch_available': TORCH_AVAILABLE,
        'tensorflow_available': TF_AVAILABLE,
        'custom_cnn_loaded': custom_cnn_model is not None,
        'mobilenet_loaded': mobilenet_model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)