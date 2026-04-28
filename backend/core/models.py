from django.db import models
import os
import uuid

def get_upload_path(instance, filename):
    return f'input/{uuid.uuid4()}_{filename}'

def get_output_path(instance, filename):
    return f'output/{uuid.uuid4()}_{filename}'

class ProcessingResult(models.Model):
    # Input image
    original_image = models.ImageField(upload_to=get_upload_path)
    
    # Output results
    dehazed_image = models.ImageField(upload_to=get_output_path, null=True, blank=True)
    detection_image = models.ImageField(upload_to=get_output_path, null=True, blank=True)
    
    # Haze detection results
    is_hazy = models.BooleanField(default=False)
    haze_confidence = models.FloatField(default=0.0)
    
    # Processing metrics
    processing_time = models.FloatField(default=0.0)  # in seconds
    original_psnr = models.FloatField(null=True, blank=True)
    enhanced_psnr = models.FloatField(null=True, blank=True)
    original_ssim = models.FloatField(null=True, blank=True)
    enhanced_ssim = models.FloatField(null=True, blank=True)
    
    # Object detection results
     # Object detection results
    objects_detected_before = models.IntegerField(default=0)
    objects_detected_after = models.IntegerField(default=0)
    detection_improvement = models.FloatField(default=0.0)
    
    # Detection images
    detection_original = models.ImageField(upload_to=get_output_path, null=True, blank=True)
    detection_enhanced = models.ImageField(upload_to=get_output_path, null=True, blank=True)
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Result {self.id} - Hazy: {self.is_hazy} - Confidence: {self.haze_confidence:.2f}"
    
    def calculate_improvement(self):
        if self.objects_detected_before > 0:
            self.detection_improvement = (
                (self.objects_detected_after - self.objects_detected_before) / 
                self.objects_detected_before * 100
            )
        else:
            self.detection_improvement = 0.0
        self.save()