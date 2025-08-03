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

try:
    from mmdet.apis import DetInferencer
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("MMDetection not available, falling back to DenseNet model")

# Configuration for mmdetection model
config_path = os.path.join(os.path.dirname(model_path), 'bone_codetr_config.py')
checkpoint_path = model_path  # The .pth file path

# Validate paths exist
if not os.path.exists(config_path):
    print(f"Warning: MMDetection config file not found at {config_path}")
    config_path = None
if not os.path.exists(checkpoint_path):
    print(f"Warning: Model checkpoint not found at {checkpoint_path}")
    checkpoint_path = None

# Class names for bone disease detection
CLASS_NAMES = ['depression', 'flatten', 'fracture', 'irregular_new_bone', 'spurformation']

# Load the pretrained DenseNet model (fallback)
def load_densenet_model():
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

# Load the mmdetection DINO model
def load_mmdet_model():
    if not MMDET_AVAILABLE:
        print("MMDetection not available")
        return None
    
    if not config_path or not checkpoint_path:
        print("MMDetection model files not found, using fallback model")
        return None
    
    try:
        # Set device
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the DetInferencer with the trained model
        inferencer = DetInferencer(config_path, checkpoint_path, device)
        print("MMDetection DINO model loaded successfully")
        return inferencer
    except Exception as e:
        print(f"Error loading mmdetection model: {e}")
        return None

def classify_image(image):
    """
    Classify X-ray image using mmdetection DINO model for object detection
    Falls back to DenseNet if mmdetection is not available
    """
    if MMDET_AVAILABLE:
        # Try to use mmdetection model first
        inferencer = load_mmdet_model()
        if inferencer is not None:
            try:
                # Save PIL image to temporary file for mmdetection
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image.save(tmp_file.name)
                    
                    # Run inference
                    result = inferencer(tmp_file.name, out_dir=None)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                    
                    # Process results
                    predictions = result['predictions'][0]
                    
                    if len(predictions['bboxes']) > 0:
                        # Get the detection with highest score
                        scores = predictions['scores']
                        labels = predictions['labels']
                        
                        max_score_idx = scores.argmax()
                        max_score = scores[max_score_idx]
                        detected_class_idx = labels[max_score_idx]
                        
                        # Map class index to class name
                        class_label = CLASS_NAMES[detected_class_idx] if detected_class_idx < len(CLASS_NAMES) else "Unknown"
                        class_probability = float(max_score)
                        
                        # Determine if healthy or not based on detections
                        if max_score > 0.5:  # Confidence threshold
                            health_status = "Not Healthy"
                            final_label = f"Detected: {class_label}"
                        else:
                            health_status = "Healthy"
                            final_label = "Healthy"
                        
                        return final_label, class_probability
                    else:
                        # No detections found - likely healthy
                        return "Healthy", 0.95
                        
            except Exception as e:
                print(f"Error in mmdetection inference: {e}")
                # Fall back to DenseNet
                pass
    
    # Fallback to original DenseNet model
    model = load_densenet_model()
    image_tensor = eval_transforms(image).unsqueeze(0)
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
        print("AI result obtained!")
        # Redirect back to the "display_xrays" view
        return redirect("display_xrays")