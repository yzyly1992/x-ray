from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import *
from PIL import Image, ImageFilter
from django.core.files.storage import default_storage
from io import BytesIO
from django.core.files.base import ContentFile
from .models import XrayImage
import os
import torch
from torchvision import transforms
from .models import XrayImage
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
from decouple import config

# Create your views here.
model_path = config('MODEL_PATH')

def success_upload(request):

    if request.method == 'GET' and 'view_btn' in request.GET:
        return redirect("display_xrays")

    if request.method == 'GET' and 'upload_btn' in request.GET:
        return redirect("xray_upload")

    return render(request, 'success_upload.html')


def display_xrays(request):

    if request.method == 'GET':
        Xrays = XrayImage.objects.all()
        return render(request, 'display_xrays.html', {'xray_img': Xrays})

    if request.method == 'POST' and 'sharpen_btn' in request.POST:
        sharpen_images(request)
        Xrays = XrayImage.objects.all()
        return render(request, 'display_xrays.html', {'xray_img': Xrays})

    if request.method == 'POST' and 'view_btn' in request.POST:
        xray_id = request.POST.get('view_btn')
        xray = XrayImage.objects.get(id=xray_id)
        return render(request, 'view_xray.html', {'xray': xray})

    if request.method == 'POST' and 'contour_btn' in request.POST:
        contour_images(request)
        Xrays = XrayImage.objects.all()
        return render(request, 'display_xrays.html', {'xray_img': Xrays})

    if request.method == 'POST' and 'delete_btn' in request.POST:
        xray_id = request.POST.get('delete_btn')
        xray = XrayImage.objects.get(id=xray_id)
        xray.xray_img.delete()
        xray.delete()
        Xrays = XrayImage.objects.all()
        return render(request, 'display_xrays.html', {'xray_img': Xrays})
    
    if request.method == 'POST' and 'sharpen_one_btn' in request.POST:
        xray_id = request.POST.get('sharpen_one_btn')
        sharpen_one_image(request, xray_id)
        Xrays = XrayImage.objects.all()
        return render(request, 'display_xrays.html', {'xray_img': Xrays})
    
    if request.method == 'POST' and 'contour_one_btn' in request.POST:
        xray_id = request.POST.get('contour_one_btn')
        contour_one_image(request, xray_id)
        Xrays = XrayImage.objects.all()
        return render(request, 'display_xrays.html', {'xray_img': Xrays})

    if request.method == 'POST':
        # Handle user feedback for each XrayImage
        for xray in XrayImage.objects.all():
            feedback = request.POST.get(f'feedback_{xray.id}')
            if feedback == 'healthy':
                xray.user_feedback = True
            elif feedback == 'abnormal':
                xray.user_feedback = False
            xray.save()
            new_data, new_feedback = prepare_new_data_and_feedback(xray.xray_img, feedback)
            # Perform retraining here
            if new_data is not None and new_feedback is not None:
                # Perform retraining here
                retrain_model(new_data, new_feedback)
        return redirect('display_xrays')
    

# Define data transforms for evaluation (similar to training transforms)
eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def sharpen_images(request):
    directory_name = "sharpened_images"

    Xrays = XrayImage.objects.all()

    for xray in Xrays:
        orig_path = xray.xray_img.path
        orig_name = os.path.basename(orig_path)
        new_path = os.path.join(directory_name, orig_name)

        img = Image.open(orig_path)
        img = img.filter(ImageFilter.SHARPEN)

        img_io = BytesIO()
        img.save(img_io, format='PNG')
        image_content = ContentFile(img_io.getvalue())

        xray.xray_img.save(new_path, image_content)

    return render(request, 'display_xrays.html', {'xray_img': Xrays})


def delete_xray(request, xray_id):
    xray = XrayImage.objects.get(id=xray_id)
    xray.xray_img.delete()
    xray.delete()
    return redirect('display_xrays')


def view_xray(request, xray_id):
    xray = XrayImage.objects.get(id=xray_id)
    return render(request, 'view_xray.html', {'xray': xray})


def contour_images(request):
    directory_name = "contoured_images"

    Xrays = XrayImage.objects.all()

    for xray in Xrays:
        img = Image.open(xray.xray_img.path)
        img = img.filter(ImageFilter.CONTOUR)

        img_io = BytesIO()
        img.save(img_io, format='PNG')
        image_content = ContentFile(img_io.getvalue())

        # Extract the filename with extension and append it to the directory name
        file_name_ext = os.path.basename(xray.xray_img.name)
        new_path = os.path.join(directory_name, file_name_ext)

        xray.xray_img.save(new_path, image_content)

    return render(request, 'display_xrays.html', {'xray_img': Xrays})

def sharpen_one_image(request, xray_id):
    directory_name = "sharpened_images"

    xray = XrayImage.objects.get(id=xray_id)
    
    img = Image.open(xray.xray_img.path)
    img = img.filter(ImageFilter.SHARPEN)

    img_io = BytesIO()
    img.save(img_io, format='PNG')
    image_content = ContentFile(img_io.getvalue())

    # Extract the filename with extension and append it to the directory name
    file_name_ext = os.path.basename(xray.xray_img.name)
    new_path = os.path.join(directory_name, file_name_ext)

    xray.xray_img.save(new_path, image_content)

    Xrays = XrayImage.objects.all()

    return render(request, 'display_xrays.html', {'xray_img': Xrays})

def contour_one_image(request, xray_id):
    directory_name = "contoured_images"

    xray = XrayImage.objects.get(id=xray_id)
    
    img = Image.open(xray.xray_img.path)
    img = img.filter(ImageFilter.CONTOUR)

    img_io = BytesIO()
    img.save(img_io, format='PNG')
    image_content = ContentFile(img_io.getvalue())

    # Extract the filename with extension and append it to the directory name
    file_name_ext = os.path.basename(xray.xray_img.name)
    new_path = os.path.join(directory_name, file_name_ext)

    xray.xray_img.save(new_path, image_content)

    Xrays = XrayImage.objects.all()

    return render(request, 'display_xrays.html', {'xray_img': Xrays})

def delete_all_xrays(request):
    Xrays = XrayImage.objects.all()
    for xray in Xrays:
        xray.xray_img.delete()
        xray.delete()
    return redirect('display_xrays')

import torch
from torchvision import models, transforms
from PIL import Image

# Load the pretrained DenseNet model and its saved state dictionary
def load_model():
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def classify_image(image):
    model = load_model()  # Load the model here
    image_tensor = eval_transforms(image).unsqueeze(0)  # Preprocess the image and add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.softmax(output, dim=1)
    class_index = predicted.item()
    class_probability = probability[0, class_index].item()
    class_label = "Healthy" if class_index == 1 else "Not Healthy"
    return class_label, class_probability

def xray_upload(request):
    if request.method == 'POST':
        form = XrayForm(request.POST, request.FILES)

        if form.is_valid():
            # xray_instance = form.save()
            # image_path = xray_instance.xray_img.path
            # img = Image.open(image_path)
            # class_label, class_probability = classify_image(img)
            # xray_instance.prediction = class_label
            # xray_instance.probability = round(class_probability,4) * 100

            xray_instance = form.save(commit=False)

            # Resize the image to 1:1 ratio and keep the center
            image = xray_instance.xray_img
            img = Image.open(image)
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            img = img.crop((left, top, right, bottom))

            # Convert the resized image back to InMemoryUploadedFile
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            image_file = InMemoryUploadedFile(
                buffer, None, image.name, 'image/png', buffer.tell(), None
            )

            # Save the resized image to the XrayImage instance
            xray_instance.xray_img = image_file
            xray_instance.prediction = "NaN"  # Set initial prediction as NaN
            xray_instance.probability = 0
            xray_instance.save()
            return redirect('success_upload')
    else:
        form = XrayForm()
    return render(request, 'image_upload.html', {'form': form})

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

def retrain_model(new_data, new_feedback):
    if new_data == None:
        pass
    # Load the pretrained DenseNet model and reset the final fully connected layer
    model = models.densenet121(pretrained=False)  # Set pretrained=False to avoid conflicts
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )

    # Load the old model.pth
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Set model to training mode and move it to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Set the model to train mode
    model.train()

    # Load data and feedback into DataLoader
    dataset = torch.utils.data.TensorDataset(new_data, torch.tensor([new_feedback]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Train for one epoch on the new data
    for epoch in range(1):
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("got the new model!!!!!!!!!!!!!")
    # Save the best model weights to model.pth
    torch.save(model.state_dict(), model_path)

def prepare_new_data_and_feedback(new_image, new_feedback):
    # Check if the new image and feedback are available
    if new_image and new_feedback:

        # Preprocess the new image
        eval_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(new_image)
        image_tensor = eval_transforms(image).unsqueeze(0)  # Preprocess the image and add batch dimension

        # Convert the feedback to 1 (Healthy) or 0 (Not Healthy)
        new_feedback = 1 if new_feedback == 'healthy' else 0

        # Return the new data and feedback
        return image_tensor, new_feedback

    # If new data or feedback is not available, return None
    return None, None

def get_ai_result(request):
    if request.method == "POST":
        xray_id = request.POST.get("get_ai_result_btn")
        xray = XrayImage.objects.get(id=xray_id)
        img = Image.open(xray.xray_img.path)
        class_label, class_probability = classify_image(img)

        # Update the XrayImage instance with the new AI result
        xray.prediction = class_label
        xray.probability = round(class_probability, 4) * 100
        xray.save()
        print("got AI result!!!")
        # Redirect back to the "display_xrays" view
        return redirect("display_xrays")