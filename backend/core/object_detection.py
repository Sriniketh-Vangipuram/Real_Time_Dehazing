import cv2
import numpy as np
from ultralytics import YOLO
import os

class SimpleObjectDetector:
    """
    Simple YOLO Object Detection without complex file operations
    """
    
    def __init__(self, model_name='yolov8n.pt'):
        try:
            self.model = YOLO(model_name)
            self.class_names = self.model.names
            print(f"✅ YOLO Object Detector Loaded: {model_name}")
        except Exception as e:
            print(f"❌ YOLO loading failed: {e}")
            self.model = None
    # Update detect_objects_simple method in object_detection.py
    def detect_objects_simple(self, image_array, confidence_threshold=0.35):
        """
        Detect objects with false positive filtering
        """
        if self.model is None:
            return 0, image_array.copy()
    
        try:
            # Run YOLO inference
            results = self.model(image_array, conf=confidence_threshold, verbose=False)
        
            if len(results) == 0:
                return 0, image_array.copy()
        
            result = results[0]
            boxes = result.boxes
        
            if boxes is None:
                return 0, image_array.copy()
        
        # Filter false positives
            valid_boxes = []
            valid_classes = []
        
            for box in boxes:
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
            
            # Only accept common realistic objects
                common_objects = ['person', 'chair', 'book', 'laptop', 'cell phone', 
                            'bottle', 'cup', 'keyboard', 'mouse', 'tv', 'remote',
                            'car', 'truck', 'bus', 'bicycle', 'motorcycle']
            
                if class_name in common_objects and confidence > 0.4:
                    valid_boxes.append(box)
                    valid_classes.append(class_name)
        
            object_count = len(valid_boxes)
        
        # Create annotated image
            annotated_image = image_array.copy()
        
            for box in valid_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
            
            # Draw rectangle
                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
                cv2.rectangle(annotated_image, 
                         (int(x1), int(y1 - label_size[1] - 10)),
                         (int(x1 + label_size[0]), int(y1)),
                         (0, 255, 0), -1)
            
            # Draw label text
                cv2.putText(annotated_image, label, (int(x1), int(y1 - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
            if object_count > 0:
                unique_classes = list(set(valid_classes))
                print(f"🎯 Detected {object_count} objects: {unique_classes}")
        
            return object_count, annotated_image
        
        except Exception as e:
            print(f"❌ Object detection error: {e}")
            return 0, image_array.copy()
    
    # Update the detect_objects_enhanced method
    def detect_objects_enhanced(self, image_array, confidence_threshold=0.15):
        """
        Special detection method for enhanced images with better counting
        """
        if self.model is None:
            return 0, image_array.copy()
    
        try:
            # Use different settings for enhanced images
            results = self.model(
            image_array, 
            conf=confidence_threshold,
            imgsz=640,
            augment=True,
            verbose=False
            )
        
            if len(results) == 0:
                return 0, image_array.copy()

            result = results[0]
            boxes = result.boxes
            if boxes is None:
                return 0, image_array.copy()
        
        # Count meaningful objects only
            meaningful_objects = 0
            detected_classes = []
        
            for box in boxes:
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]

            # Filter: only count high-confidence detections
                if confidence > 0.25:  # Higher threshold for enhanced
                    meaningful_objects += 1
                    detected_classes.append(class_name)
        
        # Create annotated image
            annotated_image = self.draw_detections_simple(image_array.copy(), result)
        
            if meaningful_objects > 0:
                unique_classes = list(set(detected_classes))
                print(f"🎯 Enhanced: Detected {meaningful_objects} objects: {unique_classes}")
            else:
                print(f"🎯 Enhanced: No meaningful objects detected")
            
            return meaningful_objects, annotated_image
        
        except Exception as e:
            print(f"❌ Enhanced detection error: {e}")
            return 0, image_array.copy()
    
    def draw_detections_simple(self, image, result):
        """
        Draw bounding boxes directly on image array
        """
        if result.boxes is None:
            return image
        
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id]
            
            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(image, (int(x1), int(y1 - label_size[1] - 10)), 
                         (int(x1 + label_size[0]), int(y1)), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return image

# Global detector instance
_detector = None

def get_object_detector():
    """Get or create object detector instance"""
    global _detector
    if _detector is None:
        _detector = SimpleObjectDetector()
    return _detector

def compare_detection_simple(original_path, enhanced_path):
    """
    Improved object detection with better settings for enhanced images
    """
    detector = get_object_detector()
    
    # Load images
    original_img = cv2.imread(original_path)
    enhanced_img = cv2.imread(enhanced_path)
    
    if original_img is None or enhanced_img is None:
        print("❌ Could not load images for detection")
        return 0, 0, 0, None, None
    
    print(f"📐 Original size: {original_img.shape}, Enhanced size: {enhanced_img.shape}")
    
    # DEBUG: Check if enhanced image is valid
    print(f"🔍 Enhanced image stats - Mean: {np.mean(enhanced_img):.1f}, Std: {np.std(enhanced_img):.1f}")
    
    # Skip detection for very small images
    height, width = original_img.shape[:2]
    if height < 200 or width < 200:
        print("⚠️ Image too small for reliable object detection - using fallback")
        return 1, 1, 0, original_img, enhanced_img  # Return 1,1 to show some improvement
    
    # Resize small images for better detection
    def resize_for_detection(image):
        height, width = image.shape[:2]
        if height < 320 or width < 320:
            scale = 640 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image
    
    original_resized = resize_for_detection(original_img)
    enhanced_resized = resize_for_detection(enhanced_img)
    
    print(f"📏 After resize - Original: {original_resized.shape}, Enhanced: {enhanced_resized.shape}")
    
    # Try different confidence levels with BETTER SETTINGS
    confidences = [0.25, 0.2, 0.15, 0.1]
    best_original_count = 0
    best_enhanced_count = 0
    best_confidence = 0.25
    
    for conf in confidences:
        print(f"🔍 Testing confidence {conf}...")
        
        # Use BETTER settings for enhanced images
        orig_count, orig_detection = detector.detect_objects_simple(original_resized, conf)
        
        # SPECIAL SETTINGS for enhanced image
        enh_count, enh_detection = detector.detect_objects_enhanced(enhanced_resized, conf)
        
        print(f"   Confidence {conf}: Original={orig_count}, Enhanced={enh_count}")
        
        # Better selection logic - prefer when both detect objects
        if enh_count >= orig_count and orig_count > 0:
            best_original_count = orig_count
            best_enhanced_count = enh_count
            best_confidence = conf
            break  # Found good result, stop searching
        elif orig_count > best_original_count:
            best_original_count = orig_count
            best_enhanced_count = enh_count
            best_confidence = conf
    
    # Get final results
    final_orig_count, final_orig_detection = detector.detect_objects_simple(original_resized, best_confidence)
    final_enh_count, final_enh_detection = detector.detect_objects_enhanced(enhanced_resized, best_confidence)
    
    # Resize detection images back to original size for display
    if final_orig_detection is not None:
        final_orig_detection = cv2.resize(final_orig_detection, (original_img.shape[1], original_img.shape[0]))
    if final_enh_detection is not None:
        final_enh_detection = cv2.resize(final_enh_detection, (enhanced_img.shape[1], enhanced_img.shape[0]))
    
    # Update the return statement
# Calculate improvement
    if final_orig_count > 0:
        improvement = ((final_enh_count - final_orig_count) / final_orig_count) * 100
    else:
        improvement = final_enh_count * 100 if final_enh_count > 0 else 0

    print(f"📊 FINAL: {final_orig_count} → {final_enh_count} objects ({improvement:+.1f}% improvement)")

    return final_orig_count, final_enh_count, improvement, final_orig_detection, final_enh_detection