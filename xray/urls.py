from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *
 
urlpatterns = [
    path('xray/upload', xray_upload, name='xray_upload'),
    path('xray/view', display_xrays, name='display_xrays'),
    path('delete_xray/<int:xray_id>', delete_xray, name='delete_xray'),
    path('view_xray/<int:xray_id>', view_xray, name='view_xray'),
    path('success-upload', success_upload, name='success_upload'),
    path('delete_all_xrays', delete_all_xrays, name='delete_all_xrays'),
    path('get_ai_result', get_ai_result, name='get_ai_result'),
    path('restore_original', restore_original_image, name='restore_original')
]
 
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


