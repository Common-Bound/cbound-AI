from django.urls import path
from . import views

urlpatterns = [
    path('recognition/', views.recognition, name='recognition'),
    path('detection/', views.detection, name='detection'),
    path('compare_string/', views.compare_string, name='compare_word'),
    path('recognition_process/', views.recognition_process,
         name='recognition_process'),
    path('recognition_thread/', views.recognition_thread,
         name='recognition_thread'),
    path('detection_thread/', views.detection_thread, name='detection_thread'),

]
