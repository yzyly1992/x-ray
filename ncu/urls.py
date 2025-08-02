"""ncu URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include
from django.conf import settings
from django.conf.urls.static import static

from django.contrib.auth.decorators import login_required
from . import views
from users.views import (
    LoginView,
    Home,
    RegisterView

)
from products.views import (
    CreateCheckoutSessionView,
    ProductLandingPageView,
    SuccessView,
    CancelView,
    stripe_webhook
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path('products/', ProductLandingPageView.as_view(), name='landing-page'),
    path('create-checkout-session/<pk>/', CreateCheckoutSessionView.as_view(), name='create-checkout-session'),
    path('webhooks/stripe/', stripe_webhook, name='stripe-webhook'),
    path('success/', SuccessView.as_view(), name='success'),
    path('cancel/', CancelView.as_view(), name='cancel'),
    path("accounts/", include("django.contrib.auth.urls")),
    path("", include("users.urls")),
    path('accounts/login/', LoginView.as_view(), name='login'),
    path('accounts/register/', RegisterView.as_view(), name='register'),
    path('', login_required(Home.as_view()), name='user_home'),
    path('', include("xray.urls"))
]

