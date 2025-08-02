from django.shortcuts import render
from django.views.generic import TemplateView
from django.urls import reverse
from django.shortcuts import redirect
from django.views import View
from django.contrib.auth import authenticate, login

# Create your views here.

"""
As you can see, most views here are commented out. This is because while they were initially created here,
they are designed to be used in the users app, which is why they are commented out and moved to users/views.py. 
The comments are left here for reference.
"""

# class Login(TemplateView):
#     template_name = './registration/login.html'

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['title'] = 'Login'
#         return context

#     def get(self, request, *args, **kwargs):
#         return render(request, self.template_name, self.get_context_data())
    
# class LoginView(View):
#     def post(self, request):
#         username = request.POST['username']
#         password = request.POST['password']
#         user = authenticate(request, username=username, password=password)
#         if user is not None:
#             login(request, user)
#             return redirect(reverse('user_home', kwargs={'username': request.user.username}))
#         else:
#             return redirect(reverse('login'))

# class Signup(TemplateView):
#     template_name = './registration/signup.html'

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['title'] = 'Signup'
#         return context

class SignUpView(View):
    def post(self, request):
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']
        user = user.objects.create_user(username, email, password) 
        user.save()
        return redirect(reverse('login'))

# class Home(TemplateView):
#     template_name = 'home.html'

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['title'] = 'Home'
#         return context

#     def get(self, request, *args, **kwargs):
#         return render(request, self.template_name, self.get_context_data())
    
class UserHomeView(TemplateView):
    template_name = 'home.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['username'] = self.request.user.username
        # add other context data as needed
        return context
    