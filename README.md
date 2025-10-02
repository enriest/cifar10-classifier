# CIFAR-10 Classifier - Lightweight Railway Deployment

A Flask web application for image classification using a Custom CNN (PyTorch) model.
Optimized for Railway deployment with minimal size and dependencies.

## 🚀 Live Demo
Deploy this app on Railway using the `deployment-clean` branch.

## 📋 Features
- Lightweight Custom CNN model (PyTorch only)
- Web interface for image upload and classification
- Real-time predictions for CIFAR-10 categories
- Bootstrap-styled responsive UI
- Fast deployment with minimal dependencies

## 🏗️ Deployment Setup

### Railway Deployment
1. Connect your Railway account to this GitHub repository
2. Use the `deployment-clean` branch for deployment
3. Railway will automatically detect the configuration from:
   - `railway.json` - Railway-specific settings
   - `Procfile` - Process configuration
   - `requirements.txt` - Python dependencies
   - `runtime.txt` - Python version

### Environment Configuration
The app is configured to run with:
- **Builder**: Nixpacks (auto-detected)
- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300`
- **Health Check**: `/health`

## 📁 Project Structure
```
├── app.py              # Main Flask application (lightweight)
├── best_model_cifar10.pth  # PyTorch model file
├── requirements.txt    # Python dependencies (PyTorch only)
├── runtime.txt        # Python version specification
├── Procfile           # Process configuration
├── railway.json       # Railway deployment settings
├── templates/         # HTML templates
│   ├── index.html     # Upload interface
│   └── results.html   # Results display
├── static/            # Static assets
│   └── uploads/       # Uploaded images directory
└── .gitignore        # Git ignore rules
```

## 🎯 CIFAR-10 Classes
The model can classify images into 10 categories:
- Airplane ✈️
- Automobile 🚗
- Bird 🐦
- Cat 🐱
- Deer 🦌
- Dog 🐶
- Frog 🐸
- Horse 🐴
- Ship 🚢
- Truck 🚛

## 💡 Model Information
- **Custom CNN**: Lightweight convolutional neural network
- **Framework**: PyTorch 2.0.1
- **Input Size**: 32x32 RGB images
- **Architecture**: 6 conv layers + 2 fully connected layers
- **Optimization**: Designed for fast inference and deployment

## 🔧 Local Development
```bash
pip install -r requirements.txt
python app.py
```

## 📊 Performance
The model is optimized for:
- ✅ Fast deployment on Railway
- ✅ Minimal memory footprint  
- ✅ Quick inference time
- ✅ Reliable predictions on CIFAR-10 categories

## 🚀 API Usage
```bash
# Upload and classify an image
curl -X POST -F "file=@image.jpg" https://your-app.railway.app/api/predict
```

## 🛠️ Technical Details
- **Python Version**: 3.11.6
- **Framework**: Flask 2.3.3
- **ML Framework**: PyTorch 2.0.1 + TorchVision 0.15.2
- **Deployment**: Railway with Nixpacks
- **Model Size**: ~2MB (optimized for cloud deployment)

---
**Note**: This is a lightweight deployment optimized for Railway. The full project with multiple models and datasets is available in the `main` branch.