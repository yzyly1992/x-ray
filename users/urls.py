"""Defines URL patterns for users."""

from django.urls import path
from .views import Login, Home
from django.contrib.auth.decorators import login_required
# from django.contrib.auth.decorators import login_required
# from .views import UserHomeView
# imports moved to ncu/urls.py for the sake of simplicity, left here for reference

app_name = 'users'

urlpatterns = [
    path('home/', login_required(Home.as_view()), name='home'),
    # path('<str:username>/home/', login_required(UserHomeView.as_view()), name='user_home'),
    # above line was moved to ncu/urls.py for the sake of simplicity, left here for reference and connected to the above imports
]