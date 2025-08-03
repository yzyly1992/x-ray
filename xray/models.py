from django.db import models

# Create your models here.
class XrayImage(models.Model):
    """
    1. Creates a new model using the models.Model builtin in django
    2. Creates the column of xray owner which denotes the user that uploaded the xray
    3. Creates the column of xray image which uses the built in ImageField method in django and uploads the image to the /images/ directory
    4. Added xray_img_origin to backup the original uploaded image
    5. Added xray_img_det to save the object detection result image
    """
    xray_owner = models.CharField(max_length=100)
    xray_img = models.ImageField(upload_to='./images/')
    xray_img_origin = models.ImageField(upload_to='./images/original/', blank=True, null=True)
    xray_img_det = models.ImageField(upload_to='./images/detection/', blank=True, null=True)
    prediction = models.CharField(max_length=50, blank=True, null=True)
    probability = models.FloatField(blank=True, null=True)
    user_feedback = models.BooleanField(default=False)

    def __str__(self):
        return self.xray_img.name
    
    def has_original(self):
        """Check if original image exists"""
        return bool(self.xray_img_origin and self.xray_img_origin.name)
    
    def has_detection_result(self):
        """Check if detection result image exists"""
        return bool(self.xray_img_det and self.xray_img_det.name)
    
    def can_restore_original(self):
        """Check if original image can be restored"""
        return self.has_original()