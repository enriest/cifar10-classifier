# CIFAR-10 Classifier - Railway Deployment![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)



A Flask web application for image classification using two different deep learning models:# Project I | Deep Learning: Image Classification with CNN

- **Custom CNN** (PyTorch) - Trained from scratch

- **MobileNetV2** (TensorFlow) - Transfer learning approach## Task Description



## 🚀 Live DemoStudents will build a Convolutional Neural Network (CNN) model to classify images from a given dataset into predefined categories/classes.

Deploy this app on Railway using the `deployment-clean` branch.

## Datasets (pick one!)

## 📋 Features

- Dual model architecture support1. The dataset for this task is the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. You can download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

- Web interface for image upload and classification2. The second dataset contains about 28,000 medium quality animal images belonging to 10 categories: dog, cat, horse, spyder, butterfly, chicken, sheep, cow, squirrel, elephant. The link is [here](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data).

- Real-time predictions for CIFAR-10 categories

- Model comparison capabilities## Assessment Components

- Bootstrap-styled responsive UI

1. **Data Preprocessing**

## 🏗️ Deployment Setup   - Data loading and preprocessing (e.g., normalization, resizing, augmentation).

   - Create visualizations of some images, and labels.

### Railway Deployment

1. Connect your Railway account to this GitHub repository2. **Model Architecture**

2. Use the `deployment-clean` branch for deployment   - Design a CNN architecture suitable for image classification.

3. Railway will automatically detect the configuration from:   - Include convolutional layers, pooling layers, and fully connected layers.

   - `railway.json` - Railway-specific settings

   - `Procfile` - Process configuration3. **Model Training**

   - `requirements.txt` - Python dependencies   - Train the CNN model using appropriate optimization techniques (e.g., stochastic gradient descent, Adam).

   - Utilize techniques such as early stopping to prevent overfitting.

### Environment Configuration

The app is configured to run with:4. **Model Evaluation**

- **Builder**: Nixpacks (auto-detected)   - Evaluate the trained model on a separate validation set.

- **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`   - Compute and report metrics such as accuracy, precision, recall, and F1-score.

- **Health Check**: Root path `/`   - Visualize the confusion matrix to understand model performance across different classes.



## 📁 Project Structure5. **Transfer Learning**

```    - Evaluate the accuracy of your model on a pre-trained models like ImagNet, VGG16, Inception... (pick one an justify your choice)

├── app.py              # Main Flask application        - You may find this [link](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub) helpful.

├── requirements.txt    # Python dependencies        - [This](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) is the Pytorch version.

├── Procfile           # Process configuration    - Perform transfer learning with your chosen pre-trained models i.e., you will probably try a few and choose the best one.

├── railway.json       # Railway deployment settings

├── templates/         # HTML templates5. **Code Quality**

│   ├── index.html     # Upload interface   - Well-structured and commented code.

│   └── results.html   # Results display   - Proper documentation of functions and processes.

├── static/            # Static assets   - Efficient use of libraries and resources.

│   └── uploads/       # Uploaded images directory

└── .gitignore        # Git ignore rules6. **Report**

```   - Write a concise report detailing the approach taken, including:

     - Description of the chosen CNN architecture.

## 🎯 CIFAR-10 Classes     - Explanation of preprocessing steps.

The models can classify images into 10 categories:     - Details of the training process (e.g., learning rate, batch size, number of epochs).

- Airplane ✈️     - Results and analysis of models performance.

- Automobile 🚗     - What is your best model. Why?

- Bird 🐦     - Insights gained from the experimentation process.

- Cat 🐱   - Include visualizations and diagrams where necessary.

- Deer 🦌   

- Dog 🐶 7. **Model deployment**

- Frog 🐸     - Pick the best model 

- Horse 🐴     - Build an app using Flask - Can you host somewhere other than your laptop? **+5 Bonus points if you use [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving)**

- Ship 🚢     - User should be able to upload one or multiples images get predictions including probabilities for each prediction

- Truck 🚛    



## 💡 Model Information## Evaluation Criteria

- **Custom CNN**: Lightweight architecture optimized for CIFAR-10

- **MobileNetV2**: Pre-trained model fine-tuned on CIFAR-10 data- Accuracy of the trained models on the validation set. **30 points**

- Both models support 32x32 RGB image inputs- Clarity and completeness of the report. **20 points**

- Quality of code implementation. **5 points**

## 🔧 Local Development- Proper handling of data preprocessing and models training. **30 points**

```bash- Demonstration of understanding key concepts of deep learning. **5 points**

pip install -r requirements.txt- Model deployment. **10 points**

python app.py

``` <span style="color:red; weight: bold">**Passing Score is 70 points**</span>.



## 📊 Performance## Submission Details

Both models are optimized for accuracy and deployment efficiency, with careful consideration for cloud hosting constraints.

- Deadline for submission: end of the week or as communicated by your teaching team.

---- Submit the following:

**Note**: This deployment branch contains only essential files. Model weights are downloaded/loaded as needed during deployment.  1. Python code files (`*.py`, `ipynb`) containing the model implementation and training process.
  2. A data folder with 5-10 images to test the deployed model/app if hosted somewhere else other than your laptop (strongly recommended! Not a must have)
  2. A PDF report documenting the approach, results, and analysis.
  3. Any additional files necessary for reproducing the results (e.g., requirements.txt, README.md).
  4. PPT presentation

## Additional Notes

- Students are encourage to experiment with different architectures, hyper-parameters, and optimization techniques.
- Provide guidance and resources for troubleshooting common issues during model training and evaluation.
- Students will discuss their approaches and findings in class during assessment evaluation sessions.

