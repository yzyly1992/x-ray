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
import shutil
import numpy as np
from .models import XrayImage
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
from decouple import config

# Create your views here.
model_path = config('MODEL_PATH', default=None)

# Convert to absolute path if it's a relative path
if model_path and not os.path.isabs(model_path):
    # Assume it's relative to the project root (where manage.py is located)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, model_path)

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
        # Delete all associated image files
        if xray.xray_img:
            xray.xray_img.delete()
        if xray.xray_img_origin:
            xray.xray_img_origin.delete()
        if xray.xray_img_det:
            xray.xray_img_det.delete()
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
    # Delete all associated image files
    if xray.xray_img:
        xray.xray_img.delete()
    if xray.xray_img_origin:
        xray.xray_img_origin.delete()
    if xray.xray_img_det:
        xray.xray_img_det.delete()
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
        # Delete all associated image files
        if xray.xray_img:
            xray.xray_img.delete()
        if xray.xray_img_origin:
            xray.xray_img_origin.delete()
        if xray.xray_img_det:
            xray.xray_img_det.delete()
        xray.delete()
    return redirect('display_xrays')

import torch
from PIL import Image

try:
    from mmdet.apis import DetInferencer
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("MMDetection not available, model inference will not work")

# Configuration for mmdetection model
config_path = None
checkpoint_path = None

if model_path:
    config_path = os.path.join(os.path.dirname(model_path), 'bone_codetr_config.py')
    checkpoint_path = model_path  # The .pth file path
else:
    print("Warning: MODEL_PATH not configured, MMDetection will not be available")

# Validate paths exist
if config_path and not os.path.exists(config_path):
    print(f"Warning: MMDetection config file not found at {config_path}")
    config_path = None
if checkpoint_path and not os.path.exists(checkpoint_path):
    print(f"Warning: Model checkpoint not found at {checkpoint_path}")
    checkpoint_path = None

# Class names for bone disease detection
CLASS_NAMES = ['depression', 'flatten', 'fracture', 'irregular_new_bone', 'spurformation']



# Load the mmdetection DINO model
def load_mmdet_model():
    if not MMDET_AVAILABLE:
        print("MMDetection not available")
        return None
    
    if not config_path or not checkpoint_path:
        print("MMDetection model files not found, using fallback model")
        return None
    
    # Additional check to ensure paths are not None and files exist
    if config_path is None or checkpoint_path is None:
        print("MMDetection model paths are None, using fallback model")
        return None
        
    if not os.path.exists(config_path) or not os.path.exists(checkpoint_path):
        print("MMDetection model files do not exist, using fallback model")
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

def classify_image_with_detection(image, return_visualization=False):
    """
    Classify X-ray image using mmdetection DINO model for object detection
    Returns classification results and optionally the visualization image
    """
    if not MMDET_AVAILABLE:
        if return_visualization:
            return "Error: MMDetection not available", 0.0, None
        return "Error: MMDetection not available", 0.0
    
    # Try to use mmdetection model
    inferencer = load_mmdet_model()
    if inferencer is None:
        if return_visualization:
            return "Error: Could not load model", 0.0, None
        return "Error: Could not load model", 0.0
    
    try:
        # Resize image to have max dimension of 1280 pixels for efficient detection
        max_dimension = 1280
        width, height = image.size
        
        if max(width, height) > max_dimension:
            # Calculate the scaling factor to maintain aspect ratio
            scale_factor = max_dimension / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize the image using LANCZOS for high quality (fallback to BICUBIC if not available)
            try:
                image = image.resize((new_width, new_height), Image.LANCZOS)
            except AttributeError:
                # Fallback for older Pillow versions
                image = image.resize((new_width, new_height), Image.BICUBIC)
            print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        else:
            print(f"Image size {width}x{height} is within limit, no resizing needed")
        
        # Save PIL image to temporary file for mmdetection
        import tempfile
        temp_dir = tempfile.mkdtemp()
        print(f"Created temp directory: {temp_dir}")
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False, dir=temp_dir) as tmp_file:
            image.save(tmp_file.name)
            print(f"Saved input image to: {tmp_file.name}")
            
            # Run inference with visualization if requested
            if return_visualization:
                result = inferencer(tmp_file.name, out_dir=temp_dir, no_save_pred=False)
            else:
                result = inferencer(tmp_file.name, no_save_vis=True)
            
            print("Inference completed")
            
            # Process classification results
            predictions = result['predictions'][0]
            final_label = "Healthy"
            class_probability = 0.95
            
            if len(predictions['bboxes']) > 0:
                # Get the detection with highest score
                scores = predictions['scores']
                labels = predictions['labels']
                
                # Convert to numpy array if it's a list
                if isinstance(scores, list):
                    scores = np.array(scores)
                if isinstance(labels, list):
                    labels = np.array(labels)
                
                max_score_idx = scores.argmax()
                max_score = scores[max_score_idx]
                detected_class_idx = labels[max_score_idx]
                
                # Map class index to class name
                class_label = CLASS_NAMES[detected_class_idx] if detected_class_idx < len(CLASS_NAMES) else "Unknown"
                class_probability = float(max_score)
                
                # Determine if healthy or not based on detections
                if max_score > 0.3:  # Confidence threshold
                    # final_label = f"Detected: {class_label}"
                    final_label = f"Unhealthy: {class_label}"
                else:
                    final_label = "Healthy"
            
            # Handle visualization if requested
            detection_img = None
            if return_visualization:
                # Look for the visualization file in temp_dir
                all_files = os.listdir(temp_dir)
                print(f"Files in temp directory: {all_files}")
                
                # First check for direct image files in temp_dir
                vis_files = [f for f in all_files if f.endswith('.jpg') or f.endswith('.png')]
                vis_files = [f for f in vis_files if f != os.path.basename(tmp_file.name)]
                
                print(f"Found direct visualization files: {vis_files}")
                
                if vis_files:
                    # Load the visualization image
                    vis_path = os.path.join(temp_dir, vis_files[0])
                    print(f"Loading visualization from: {vis_path}")
                    detection_img = Image.open(vis_path)
                else:
                    # Check if there's a 'vis' directory containing visualization images
                    vis_dir_path = os.path.join(temp_dir, 'vis')
                    if os.path.exists(vis_dir_path) and os.path.isdir(vis_dir_path):
                        print(f"Found vis directory: {vis_dir_path}")
                        vis_dir_files = os.listdir(vis_dir_path)
                        print(f"Files in vis directory: {vis_dir_files}")
                        
                        # Look for image files in the vis directory
                        vis_image_files = [f for f in vis_dir_files if f.endswith('.jpg') or f.endswith('.png')]
                        
                        if vis_image_files:
                            vis_image_path = os.path.join(vis_dir_path, vis_image_files[0])
                            print(f"Loading visualization from vis directory: {vis_image_path}")
                            detection_img = Image.open(vis_image_path)
                        else:
                            print("No image files found in vis directory")
                            detection_img = image  # Return original image as fallback
                    else:
                        # Check for any other output files with different patterns
                        output_files = [f for f in all_files if f != os.path.basename(tmp_file.name)]
                        if output_files:
                            print(f"No vis directory found, trying other output files: {output_files}")
                            for output_file in output_files:
                                vis_path = os.path.join(temp_dir, output_file)
                                if os.path.isfile(vis_path):
                                    try:
                                        detection_img = Image.open(vis_path)
                                        print("Successfully loaded alternative visualization file")
                                        break
                                    except Exception as e:
                                        print(f"Could not open {vis_path} as image: {e}")
                                        continue
                            
                            if detection_img is None:
                                print("No valid visualization files found, returning original image")
                                detection_img = image
                        else:
                            print("No visualization files found, returning original image")
                            detection_img = image
            
            # Clean up temporary files
            os.unlink(tmp_file.name)
            shutil.rmtree(temp_dir)
            print("Cleanup completed")
            
            if return_visualization:
                return final_label, class_probability, detection_img
            return final_label, class_probability
            
    except Exception as e:
        print(f"Error in mmdetection inference: {e}")
        import traceback
        traceback.print_exc()
        
        if return_visualization:
            return "Error: MMDetection inference failed", 0.0, None
        return "Error: MMDetection inference failed", 0.0

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
            buffer.seek(0)  # Reset buffer position
            image_file = InMemoryUploadedFile(
                buffer, None, image.name, 'image/png', buffer.tell(), None
            )

            # Save the resized image to the XrayImage instance
            xray_instance.xray_img = image_file
            
            # Also save a copy to xray_img_origin for backup
            buffer_origin = BytesIO()
            img.save(buffer_origin, format='PNG')
            buffer_origin.seek(0)  # Reset buffer position
            origin_image_file = InMemoryUploadedFile(
                buffer_origin, None, f"origin_{image.name}", 'image/png', buffer_origin.tell(), None
            )
            xray_instance.xray_img_origin = origin_image_file
            
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
        
        # Get AI classification result and visualization
        class_label, class_probability, detection_result_img = classify_image_with_detection(img, return_visualization=True)

        # Update the XrayImage instance with the new AI result
        xray.prediction = class_label
        xray.probability = round(class_probability, 4) * 100
        
        # Save result images if detection visualization was generated
        if detection_result_img:
            # Save detection result to xray_img_det
            buffer_det = BytesIO()
            detection_result_img.save(buffer_det, format='PNG')
            buffer_det.seek(0)  # Reset buffer position to beginning
            det_image_file = InMemoryUploadedFile(
                buffer_det, None, f"det_{os.path.basename(xray.xray_img.name)}", 
                'image/png', buffer_det.tell(), None
            )
            xray.xray_img_det = det_image_file
            
            # Replace xray_img with detection result
            buffer_main = BytesIO()
            detection_result_img.save(buffer_main, format='PNG')
            buffer_main.seek(0)  # Reset buffer position to beginning
            main_image_file = InMemoryUploadedFile(
                buffer_main, None, f"{os.path.basename(xray.xray_img.name)}", 
                'image/png', buffer_main.tell(), None
            )
            xray.xray_img = main_image_file
            print("Detection visualization saved successfully!")
        else:
            print("No detection visualization generated")
        
        xray.save()
        print("AI result obtained!")
        # Redirect back to the "display_xrays" view
        return redirect("display_xrays")

def restore_original_image(request):
    """
    Restore the original image to xray_img field
    """
    if request.method == "POST":
        xray_id = request.POST.get("restore_btn")
        xray = XrayImage.objects.get(id=xray_id)
        
        if xray.xray_img_origin:
            # Copy original image back to xray_img
            original_img = Image.open(xray.xray_img_origin.path)
            
            buffer = BytesIO()
            original_img.save(buffer, format='PNG')
            buffer.seek(0)  # Reset buffer position
            restored_image_file = InMemoryUploadedFile(
                buffer, None, f"restored_{os.path.basename(xray.xray_img_origin.name)}", 
                'image/png', buffer.tell(), None
            )
            xray.xray_img = restored_image_file
            xray.save()
            print("Original image restored!")
        
        return redirect("display_xrays")