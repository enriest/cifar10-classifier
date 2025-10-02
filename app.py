"""
Lightweight Flask App for CIFAR-10 Classification
Supports Custom CNN and ResNet18 (PyTorch) - optimized for Railway deployment
Reflects the models from Project.ipynb notebook
"""

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
import os
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MODEL_PATH = 'best_model_cifar10.pth'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Global model variable
model = None

# Custom CNN class definition for the lightweight model
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
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
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
        x = self.dropout3(x)
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def load_model():
    """Load the Custom CNN PyTorch model"""
    global model
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available")
        return None
    
    try:
        if os.path.exists(MODEL_PATH):
            # Try to load as checkpoint first
            try:
                checkpoint = torch.load(MODEL_PATH, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    # It's a checkpoint - determine the model architecture
                    state_dict = checkpoint['model_state_dict']
                    
                    # Check if it's ResNet18 or custom CNN based on keys
                    if any('fc.weight' in key for key in state_dict.keys()):
                        # It's ResNet18
                        model = models.resnet18(pretrained=False)
                        model.fc = torch.nn.Linear(model.fc.in_features, 10)
                    else:
                        # It's our custom CNN
                        model = CNN()
                    
                    model.load_state_dict(state_dict)
                else:
                    # Direct state dict
                    model = CNN()
                    model.load_state_dict(checkpoint)
            except:
                # Try loading as direct state dict
                model = CNN()
                model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            
            model.eval()
            print("✅ Model loaded successfully!")
            return model
        else:
            print(f"❌ No model found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_type():
    """Determine if loaded model is ResNet18 or custom CNN"""
    if model is None:
        return 'custom'
    
    # Check if model has 'fc' layer (ResNet18) or not (Custom CNN)
    for name, _ in model.named_parameters():
        if 'fc.weight' in name:
            return 'resnet'
    return 'custom'

def preprocess_image(image_file, model_type=None):
    """Preprocess uploaded image for PyTorch model"""
    try:
        # Auto-detect model type if not provided
        if model_type is None:
            model_type = get_model_type()
        
        # Open and convert to RGB
        image = Image.open(image_file).convert('RGB')
        
        if model_type == 'resnet':
            # ResNet18 preprocessing (ImageNet style, 224x224)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            # Custom CNN preprocessing (CIFAR-10 style, 32x32)
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def predict(image_tensor):
    """Make prediction using the model"""
    if model is None:
        raise Exception("Model not loaded")
    
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
        raise Exception(f"Prediction error: {str(e)}")

# Load model at startup
if TORCH_AVAILABLE:
    load_model()

@app.route('/')
def index():
    """Main upload page"""
    model_status = "✅ Loaded" if model is not None else "❌ Not Loaded"
    
    return render_template('index.html', 
                         model_status=model_status,
                         torch_available=TORCH_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handle file upload and make predictions"""
    if not TORCH_AVAILABLE:
        flash('PyTorch not available.')
        return redirect(url_for('index'))
    if model is None:
        flash('Model not loaded. Please check server logs.')
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
            # Preprocess and predict
            image_tensor = preprocess_image(file)
            results = predict(image_tensor)
            
            # Convert image to base64 for display
            file.seek(0)  # Reset file pointer
            image_b64 = base64.b64encode(file.read()).decode('utf-8')
            
            return render_template('results.html', 
                                 results=results,
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
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            image_tensor = preprocess_image(file)
            results = predict(image_tensor)
            
            return jsonify({
                'success': True,
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
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)