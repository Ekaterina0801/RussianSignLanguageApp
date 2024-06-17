"""
URL configuration for SignLanguageApp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path, include
from sign import views
from django.contrib.auth import views as au

urlpatterns = [
    path('accounts/', include('django.contrib.auth.urls')),
    path('admin/', admin.site.urls),
    path('', views.index),
    path('recognizer', views.recognizer),
    path('camera/<str:label>', views.livefe, name="show"),
    path('camera_test/<str:label>', views.livefe_test, name="show_test"),
    path('lessons_main',views.lessons_main),
    path('lesson/<str:category>',views.lesson),
    path('login/', au.LoginView.as_view(), name='login'),
    path('profile', views.profile),
    path('signup/', views.signup, name='signup'),
    path('check/<str:label>', views.checker),
    path('start_test/<str:category>', views.start_test),
    path('test/<int:number>', views.test, name='test'),
    path('tests_main', views.tests_main),
    path('finish_test', views.finish_test, name='finish_test'),
]
