"""
Advanced Flask App for CIFAR-10 Classification
Supports multiple model architectures as implemented in the Jupyter notebooks:
- Custom CNN (from Project.ipynb)
- ResNet18 Transfer Learning (from Project.ipynb)
- Multiple Transfer Learning Models (from Transfer learning.ipynb)
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
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not installed. Install with: pip install torch torchvision")

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Model paths - matching your notebook implementations
MODEL_PATHS = {
    'custom_cnn': 'models/best_model_cifar10.pth',
    'resnet18': 'models/best_transfer_model_resnet18.pth',
    'mobilenet_v2': 'models/best_transfer_model_mobilenet_v2.pth',
    'efficientnet_b0': 'models/best_transfer_model_efficientnet_b0.pth'
}

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
loaded_models = {}

# Custom CNN Architecture (from Project.ipynb)
class CNN(nn.Module):
    """
    Custom CNN architecture as implemented in Project.ipynb
    Features:
    - 6 Convolutional layers with batch normalization
    - 3 Max pooling layers with dropout
    - 2 Fully connected layers
    - Designed for CIFAR-10 (32x32x3 input)
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)

        # Third convolutional block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Fourth convolutional block
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)

        # Fifth and sixth convolutional blocks
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Calculated from conv output size
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10
    
    def forward(self, x):
        # First block: Conv -> BN -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second block: Conv -> BN -> ReLU -> Pool -> Dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Third block: Conv -> BN -> ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Fourth block: Conv -> BN -> ReLU -> Pool -> Dropout
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Fifth block: Conv -> BN -> ReLU
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Sixth block: Conv -> BN -> ReLU -> Pool -> Dropout
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

def create_transfer_model(model_name, num_classes=10):
    """
    Create transfer learning models as implemented in Transfer learning.ipynb
    Supports: ResNet18, MobileNetV2, EfficientNet-B0
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        # Replace final layer for CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Replace classifier for CIFAR-10
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Replace classifier for CIFAR-10
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def load_model(model_name):
    """Load a specific model based on the notebook implementations"""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return None
    
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if model_name == 'custom_cnn':
            # Create custom CNN model
            model = CNN()
            model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # Create transfer learning model
            model = create_transfer_model(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print(f"✅ {model_name} model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {e}")
        return None

def load_all_models():
    """Load all available models at startup"""
    for model_name in MODEL_PATHS.keys():
        model = load_model(model_name)
        if model is not None:
            loaded_models[model_name] = model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_transforms(model_name):
    """
    Get appropriate transforms for each model based on notebook implementations
    """
    if model_name == 'custom_cnn':
        # Custom CNN uses CIFAR-10 specific normalization (from Project.ipynb)
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    
    elif model_name in ['resnet18', 'mobilenet_v2', 'efficientnet_b0']:
        # Transfer learning models use ImageNet normalization and 224x224 input
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def preprocess_image(image_file, model_name):
    """Preprocess uploaded image for the specified model"""
    try:
        # Get appropriate transforms
        transform = get_transforms(model_name)
        
        # Open and convert to RGB
        image = Image.open(image_file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    except Exception as e:
        raise Exception(f"Error preprocessing image for {model_name}: {str(e)}")

def predict_with_model(image_tensor, model_name):
    """Make prediction using the specified model"""
    if model_name not in loaded_models:
        raise Exception(f"{model_name} model not loaded")
    
    model = loaded_models[model_name]
    
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
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
        raise Exception(f"{model_name} prediction error: {str(e)}")

def get_model_info(model_name):
    """Get model information as described in the notebooks"""
    model_info = {
        'custom_cnn': {
            'name': 'Custom CNN',
            'description': 'Custom 6-layer CNN with batch normalization and dropout',
            'input_size': '32x32',
            'source': 'Project.ipynb - Built from scratch',
            'features': ['6 Conv layers', 'Batch normalization', 'Dropout regularization', 'Designed for CIFAR-10']
        },
        'resnet18': {
            'name': 'ResNet18 Transfer Learning',
            'description': 'Pre-trained ResNet18 with frozen backbone, fine-tuned classifier',
            'input_size': '224x224',
            'source': 'Project.ipynb - Transfer learning from ImageNet',
            'features': ['Residual connections', 'Deep architecture', 'Transfer learning', 'ImageNet pre-training']
        },
        'mobilenet_v2': {
            'name': 'MobileNetV2 Transfer Learning',
            'description': 'Efficient mobile-optimized CNN with depthwise separable convolutions',
            'input_size': '224x224',
            'source': 'Transfer learning.ipynb - Model comparison',
            'features': ['Depthwise separable conv', 'Mobile-optimized', 'Transfer learning', 'Lightweight']
        },
        'efficientnet_b0': {
            'name': 'EfficientNet-B0 Transfer Learning',
            'description': 'Compound scaling CNN optimizing depth, width, and resolution',
            'input_size': '224x224',
            'source': 'Transfer learning.ipynb - Model comparison',
            'features': ['Compound scaling', 'Efficient architecture', 'Transfer learning', 'Balanced design']
        }
    }
    return model_info.get(model_name, {})

# Load models at startup
if TORCH_AVAILABLE:
    load_all_models()

@app.route('/')
def index():
    """Main upload page with model status"""
    model_status = {}
    for model_name in MODEL_PATHS.keys():
        model_status[model_name] = {
            'loaded': model_name in loaded_models,
            'info': get_model_info(model_name)
        }
    
    return render_template('index.html', 
                         model_status=model_status,
                         torch_available=TORCH_AVAILABLE,
                         total_models=len(MODEL_PATHS),
                         loaded_models=len(loaded_models))

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and make predictions"""
    selected_models = request.form.getlist('models')  # Allow multiple model selection
    
    if not selected_models:
        flash('Please select at least one model.')
        return redirect(url_for('index'))
    
    # Check if selected models are available
    available_models = []
    for model_name in selected_models:
        if model_name in loaded_models:
            available_models.append(model_name)
        else:
            flash(f'{model_name} model not available.')
    
    if not available_models:
        flash('No selected models are available.')
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
            
            # Make predictions with each selected model
            for model_name in available_models:
                file.seek(0)  # Reset file pointer for each model
                image_tensor = preprocess_image(file, model_name)
                model_results = predict_with_model(image_tensor, model_name)
                results[model_name] = {
                    'predictions': model_results,
                    'info': get_model_info(model_name)
                }
            
            # Convert image to base64 for display
            file.seek(0)
            image_b64 = base64.b64encode(file.read()).decode('utf-8')
            
            return render_template('results.html', 
                                 results=results,
                                 selected_models=available_models,
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
    selected_models = request.form.getlist('models')
    
    if not selected_models:
        return jsonify({'error': 'No models selected'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            results = {}
            
            for model_name in selected_models:
                if model_name in loaded_models:
                    file.seek(0)
                    image_tensor = preprocess_image(file, model_name)
                    model_results = predict_with_model(image_tensor, model_name)
                    results[model_name] = {
                        'predictions': model_results,
                        'info': get_model_info(model_name)
                    }
            
            return jsonify({
                'success': True,
                'models': list(results.keys()),
                'results': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/models')
def api_models():
    """API endpoint to get available models and their info"""
    models_info = {}
    for model_name in MODEL_PATHS.keys():
        models_info[model_name] = {
            'loaded': model_name in loaded_models,
            'info': get_model_info(model_name)
        }
    
    return jsonify({
        'available_models': models_info,
        'total_models': len(MODEL_PATHS),
        'loaded_models': len(loaded_models),
        'pytorch_available': TORCH_AVAILABLE
    })

@app.route('/health')
def health():
    """Health check endpoint for cloud platforms"""
    return jsonify({
        'status': 'healthy',
        'pytorch_available': TORCH_AVAILABLE,
        'models_loaded': len(loaded_models),
        'total_models': len(MODEL_PATHS),
        'available_models': list(loaded_models.keys())
    })

@app.route('/compare')
def compare():
    """Model comparison page showcasing notebook findings"""
    comparison_data = {
        'custom_cnn': {
            'accuracy': 'Variable (depends on training)',
            'parameters': '~2.5M',
            'input_size': '32x32',
            'training_time': 'Medium',
            'advantages': ['Designed for CIFAR-10', 'Lightweight', 'Custom architecture'],
            'notebook': 'Project.ipynb'
        },
        'resnet18': {
            'accuracy': 'High (transfer learning)',
            'parameters': '~11M',
            'input_size': '224x224', 
            'training_time': 'Fast (frozen backbone)',
            'advantages': ['Residual connections', 'Deep architecture', 'Transfer learning'],
            'notebook': 'Project.ipynb'
        },
        'mobilenet_v2': {
            'accuracy': 'High (efficient)',
            'parameters': '~3.5M',
            'input_size': '224x224',
            'training_time': 'Fast',
            'advantages': ['Mobile-optimized', 'Depthwise separable conv', 'Efficient'],
            'notebook': 'Transfer learning.ipynb'
        },
        'efficientnet_b0': {
            'accuracy': 'Very High',
            'parameters': '~5.3M',
            'input_size': '224x224',
            'training_time': 'Medium',
            'advantages': ['Compound scaling', 'State-of-the-art', 'Balanced design'],
            'notebook': 'Transfer learning.ipynb'
        }
    }
    
    return render_template('compare.html', comparison_data=comparison_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
