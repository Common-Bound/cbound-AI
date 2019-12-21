from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ocr/', include('ocr.urls')),
    path('object/', include('object.urls')),
    path('inspection/', include('inspection.urls')),
]
