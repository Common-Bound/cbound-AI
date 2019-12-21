from django.urls import path
from . import views

urlpatterns = [
    path('detection/', views.detection, name='detection'),
    path('recognition/', views.recognition, name='recognition'),
]
