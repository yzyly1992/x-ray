#!/usr/bin/env python
"""
Test script for the merged detection and classification function
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ncu.settings')
django.setup()

from xray.views import classify_image_with_detection
from PIL import Image

def test_merged_function():
    """Test the merged classification and detection function"""
    
    # Check if we have a test image
    test_images = ['media/images/0028_r0.jpg', 'media/images/0137_r0.jpg']
    test_image_path = None

    for img_path in test_images:
        if os.path.exists(img_path):
            test_image_path = img_path
            break

    if not test_image_path:
        print('No test images found')
        return False

    print(f'Testing with image: {test_image_path}')
    test_img = Image.open(test_image_path)
    
    print(f'Input image size: {test_img.size}')
    print(f'Input image mode: {test_img.mode}')
    
    # Test classification only
    print('\n--- Testing Classification Only ---')
    label, probability = classify_image_with_detection(test_img, return_visualization=False)
    print(f'Classification: {label}')
    print(f'Probability: {probability:.4f}')
    
    # Test classification with visualization
    print('\n--- Testing Classification with Visualization ---')
    label, probability, vis_img = classify_image_with_detection(test_img, return_visualization=True)
    print(f'Classification: {label}')
    print(f'Probability: {probability:.4f}')
    
    if vis_img:
        print('Visualization successful!')
        print(f'Visualization image size: {vis_img.size}')
        print(f'Visualization image mode: {vis_img.mode}')
        
        # Save the result for inspection
        vis_img.save('test_detection_result.png')
        print('Saved visualization to test_detection_result.png')
        return True
    else:
        print('Visualization failed!')
        return False

if __name__ == '__main__':
    success = test_merged_function()
    sys.exit(0 if success else 1)
