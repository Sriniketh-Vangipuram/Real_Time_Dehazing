from django.urls import path
from . import views

# app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('process/<int:result_id>/', views.process_image, name='process_image'),
    path('results/<int:result_id>/', views.view_results, name='view_results'),
    path('video/', views.video_processing, name='video_processing'),
    path('video/feed/', views.video_feed, name='video_feed'),  # NEW
    path('browse-dataset/', views.browse_dataset, name='browse_dataset'),
    path('video/history/', views.detection_history, name='detection_history'),
    path('video/history/', views.detection_history, name='detection_history'),
]