# X-Ray Examination System Technical Documentation

This technical documentation provides an overview of the X-Ray Examination System, including its architecture, functionalities, and technical details. It also explains how to work with the system's AI model, how to replace the model, and how to retrain it using user feedback.

## Table of Contents
1. Introduction
2. Architecture Overview
3. Technologies Used
4. Folder Structure
5. AI Model Details
6. Replacing the AI Model
7. Retraining the AI Model
8. Conclusion

## 1. Introduction
The X-Ray Examination System is a web application that allows users to upload X-ray images and receive AI-based evaluations of whether the X-rays indicate healthy or abnormal conditions. The system offers additional image processing options, such as sharpening and contouring. Users can also provide feedback on AI evaluations, which contributes to the retraining of the AI model for improved accuracy over time.

## 2. Architecture Overview
The application follows a client-server architecture, where the client (web browser) interacts with the server (Django-based backend) to perform various operations, including image uploads, image processing, AI evaluations, and more. The server communicates with the AI model to obtain predictions for the uploaded X-ray images.

## 3. Technologies Used
The X-Ray Examination System is developed using the following technologies:
- Django: A high-level Python web framework for rapid development and clean design.
- Python: The programming language used for the application's backend and AI-related tasks.
- HTML/CSS: Frontend technologies for the user interface.
- PIL (Python Imaging Library) and torchvision: Libraries used for image processing and handling AI models, respectively.
- PyTorch: The deep learning library used for the AI model.

## 4. Folder Structure
The main folder structure of the X-Ray Examination System is as follows:

    X-Ray-Examination-System/
├── app/ (Django app folder)
│   ├── migrations/
│   ├── static/
│   ├── templates/
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
├── CNNs/ (Folder containing AI models)
│   ├── model.pth
│   └── ... (other models, if any)
├── XRayExaminationSystem/
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── manage.py
└── ...


## 5. AI Model Details
The AI model used in the X-Ray Examination System is a DenseNet121 architecture, pretrained on a large dataset of X-ray images. The last fully connected layer of the original DenseNet model is replaced with a custom classification layer to fit the binary classification task (healthy or abnormal).

### Loading the AI Model (def load_model())
The `load_model()` function is responsible for loading the pretrained AI model from the `CNNs` folder. The function then replaces the last classification layer with a custom classification layer to adapt it for our specific task. The model is set to evaluation mode and moved to the CPU.

### AI Model Classification (def classify_image())
The `classify_image()` function takes an input image and uses the AI model to predict whether the X-ray image indicates a "Healthy" or "Not Healthy" condition. The model's prediction and probability are returned as outputs.

## 6. Replacing the AI Model
To replace the AI model with a different model, follow these steps:

1. Train or obtain the new AI model externally, ensuring it is saved as a PyTorch state dictionary file (e.g., `new_model.pth`).
2. Place the new model file in the `CNNs` folder of the application.
3. Update the `model_path` variable in the `views.py` file to point to the new model file. For example:
   ```python
   model_path = 'CNNs/new_model.pth'
    The application will now use the new AI model for image evaluations.
    
## 7. Retraining the AI Model (def retrain_model(new_data, new_feedback))
The `retrain_model()` function is responsible for retraining the AI model using new data and user feedback. When a user provides feedback on an X-ray image evaluation, the system collects the new data and feedback. The function then loads the current AI model, sets it to training mode, and trains it on the new data.

### Preparing New Data and Feedback
The `prepare_new_data_and_feedback()` function prepares the new data (X-ray image) and feedback (healthy or abnormal) provided by the user. It preprocesses the image and returns the data in a format suitable for training the AI model.

### Note on Retraining
The retraining process is triggered when users provide feedback on the AI evaluations. The system continuously improves its performance over time by incorporating user feedback and retraining the AI model.

## 8. Conclusion
The X-Ray Examination System provides an efficient and user-friendly platform for X-ray image evaluations using AI algorithms. Users can upload X-ray images, view evaluations, and provide valuable feedback for model retraining, leading to more accurate evaluations in the future.

If you encounter any issues or have questions regarding the technical aspects of the X-Ray Examination System, please refer to this documentation or contact our technical support team for assistance.
