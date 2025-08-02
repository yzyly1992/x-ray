from django.db import models

# Create your models here.
class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.IntegerField(default=0) # cents
    url = models.URLField('127.0.0.1:8000/image')

    def __str__(self):
        return self.name

    def get_display_price (self):
        return "{0:.2f}".format(self.price / 100)

if __name__ == "__main__":
    Scans = Product(name="Scans", price=0, url='127.0.0.1:8000')
    Scans.save()