"""This module contains the views for the users app"""

from django.shortcuts import render
from django.views.generic import TemplateView
from django.urls import reverse
from django.shortcuts import redirect
from django.views import View
from django.contrib.auth import authenticate, login

from django.shortcuts import render, redirect
from django.views import View
from django.contrib.auth.forms import UserCreationForm

from .forms import RegisterForm

# Create your views here.


class Home(TemplateView):
    """
    1. Creates a class that inherits from the TemplateView class
    2. Creates a template_name variable that tells Django which template should be used to render the view
    3. Creates a get_context_data method that adds extra context variables to the template
    4. Creates a get method that renders the template and passes it the context
    """
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Home'
        context['username'] = self.request.user.username
        return context

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.get_context_data())


class Login(TemplateView):
    """
    1. Creates a class that inherits from the TemplateView class
    2. Creates a template_name variable that tells Django which template should be used to render the view
    3. Creates a get_context_data method that adds extra context variables to the template
    4. Creates a get method that renders the template and passes it the context
    """
    template_name = './registration/login.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Login'
        return context

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.get_context_data())


class LoginView(View):
    """
    1. Creates a class that inherits from the View class
    2. Creates a get method that renders the template and passes it the context
    3. Creates a post method that authenticates the user and logs them in
    """
    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse('user_home', kwargs={'username': request.user.username}))
        else:
            return redirect(reverse('login'))

#RegisterForm
# ^ Custom alternative to UserCreationForm
# unnecessary for time being, but kept for posterity     

class RegisterView(View):
    """
    1. Creates a class that inherits from the View class
    2. Creates a get method that renders the template and passes it the context
    3. Creates a post method that saves the form and redirects the user to the login page
    """
    def get(self, request):
        form = RegisterForm()
        return render(request, 'registration/register.html', {'form': form})

    def post(self, request):
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
        else:
            print(form.errors)
            return render(request, 'registration/register.html', {'form': form})
        
        # kept for time being for posterity. The below code is non-functional when implemented as is.
        # if form.is_valid():
        #     user = form.save(commit=False)
        #     user.username = user.username()
        #     user.save()
        #     redirect(reverse('login'))
        # else:
        #     return render(request, './regestration/register.html', {'form': form})