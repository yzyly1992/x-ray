"""This file contains the form that will be used to register a new user"""

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    """
    1. Creates a form that inherits from the UserCreationForm class
    2. Adds an email field to the form
    3. Creates a nested class that tells Django which model should be used to create this form (model = User)
    4. Tells Django which fields should be included in the form (fields = ['username', 'email', 'password1', 'password2'])
    """
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']
