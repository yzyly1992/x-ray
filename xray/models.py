from django.db import models

# Create your models here.
class XrayImage(models.Model):
    """
    1. Creates a new model using the models.Model builtin in django
    2. Creates the column of xray owner which denotes the user that uploaded the xray
    3. Creates the column of xray image whice uses the built in ImageField method in django and uploads the image to the /images/ directory
    """
    xray_owner = models.CharField(max_length=100)
    xray_img = models.ImageField(upload_to='./images/')
    prediction = models.CharField(max_length=50, blank=True, null=True)
    probability = models.FloatField(blank=True, null=True)
    user_feedback = models.BooleanField(default=False)

    def __str__(self):
        return self.xray_img.name