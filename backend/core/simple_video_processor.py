import cv2
import numpy as np
import time
import threading
import os
from datetime import datetime

class SimpleVideoProcessor:
    def __init__(self):
        self.is_processing = False
        self.current_frame = None
        self.frame_count = 0
        self.start_time = None
        self.processing_thread = None
        
        # Initialize detector
        try:
            from .object_detection import get_object_detector
            self.detector = get_object_detector()
            print("✅ Object detector loaded")
        except Exception as e:
            print(f"⚠️ Object detector not available: {e}")
            self.detector = None
        
        # Performance tracking
        self.fps_history = []
        self.objects_history = []
        
        # For box persistence
        self.last_detection_frame = None
        self.last_detection_count = 0
        self.frames_since_detection = 0
        
        # Detection logging
        self.detection_log = []
        self.last_save_time = time.time()
        
        # Create output directory
        self.output_dir = "media/detection_logs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_processing(self, camera_index=0):
        """Start video processing"""
        if self.is_processing:
            return False
            
        self.is_processing = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history = []
        self.objects_history = []
        self.last_detection_frame = None
        self.last_detection_count = 0
        self.frames_since_detection = 0
        self.detection_log = []
        self.last_save_time = time.time()
        
        # Create session folder
        session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_dir, f"session_{session_time}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.processing_thread = threading.Thread(
            target=self._process_video_with_logging,
            args=(camera_index,),
            daemon=True
        )
        self.processing_thread.start()
        
        print(f"🎬 Video processing started - Session: {session_time}")
        return True
    
    def _process_video_with_logging(self, camera_index):
        """Process video with detection logging"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_index}")
            self.is_processing = False
            return
        
        print(f"📹 Camera opened - Saving to: {self.session_dir}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Calculate FPS
            current_fps = self.frame_count / (time.time() - self.start_time)
            self.fps_history.append(current_fps)
            
            # Process frame
            processed, objects_detected, detected_classes = self._process_frame_with_logging(frame)
            
            # Log detection if objects found
            if objects_detected > 0:
                self._log_detection(frame, processed, objects_detected, detected_classes)
            
            self.current_frame = processed
            
            # Control FPS
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        print("✅ Camera released")
    
    def _process_frame_with_logging(self, frame):
        """Process frame and return detection info"""
        processed = frame.copy()
        objects_detected = 0
        detected_classes = []
        
        # OBJECT DETECTION (every 3rd frame)
        if self.detector and self.frame_count % 3 == 0:
            object_count, detected_frame, classes = self._detect_with_logging(frame)
            
            if detected_frame is not None:
                processed = detected_frame
                objects_detected = object_count
                detected_classes = classes
                self.objects_history.append(object_count)
        
        # SIMPLE DEHAZING EFFECT (every 15th frame)
        if self.frame_count % 15 == 0:
            processed = self._apply_visual_enhancement(processed)
        
        # ADD OVERLAYS
        processed = self._add_info_overlays(processed, objects_detected, detected_classes)
        
        return processed, objects_detected, detected_classes
    
    def _detect_with_logging(self, frame):
        """Detect objects and return classes"""
        try:
            # Run detection
            object_count, detected_frame = self.detector.detect_objects_simple(
                frame, confidence_threshold=0.35
            )
            
            # Extract classes from frame (you might need to modify your detector)
            # For now, we'll parse from print statements
            detected_classes = ['person']  # Default, will be updated
            
            if detected_frame is not None and object_count > 0:
                # Update last detection
                self.last_detection_frame = detected_frame
                self.last_detection_count = object_count
                self.frames_since_detection = 0
                return object_count, detected_frame, detected_classes
            else:
                self.frames_since_detection += 1
                
                # If we had detection recently, keep showing it
                if self.last_detection_frame is not None and self.frames_since_detection < 8:
                    return self.last_detection_count, self.last_detection_frame, ['person']
                else:
                    return 0, None, []
                    
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return 0, None, []
    
    def _log_detection(self, original_frame, processed_frame, object_count, classes):
        """Save detection to log and capture screenshot"""
        try:
            current_time = datetime.now()
            timestamp = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Create detection record
            detection_record = {
                'timestamp': current_time,
                'frame_number': self.frame_count,
                'object_count': object_count,
                'classes': classes,
                'fps': self.frame_count / (time.time() - self.start_time)
            }
            
            # Save to log
            self.detection_log.append(detection_record)
            
            # Save screenshot every 30 detections or every 10 seconds
            save_interval = 10  # seconds
            if (len(self.detection_log) % 30 == 0 or 
                time.time() - self.last_save_time > save_interval):
                
                # Save original and processed frames
                original_path = os.path.join(self.session_dir, f"orig_{timestamp}.jpg")
                processed_path = os.path.join(self.session_dir, f"proc_{timestamp}.jpg")
                
                cv2.imwrite(original_path, original_frame)
                cv2.imwrite(processed_path, processed_frame)
                
                print(f"📸 Saved detection: {original_path}")
                self.last_save_time = time.time()
                
                # Update record with file paths
                detection_record['original_image'] = original_path
                detection_record['processed_image'] = processed_path
            
            return detection_record
            
        except Exception as e:
            print(f"⚠️ Error logging detection: {e}")
            return None
    
    def _apply_visual_enhancement(self, frame):
        """Apply visual enhancement to simulate dehazing"""
        try:
            # Simple contrast enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Blend with original
            result = cv2.addWeighted(frame, 0.3, enhanced, 0.7, 0)
            
            # Add enhancement indicator
            cv2.putText(result, "ENHANCED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            return result
        except:
            return frame
    
    def _add_info_overlays(self, frame, objects_detected, classes):
        """Add information overlays"""
        height, width = frame.shape[:2]
        
        # FPS counter
        current_fps = self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0
        fps_text = f"FPS: {current_fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Objects counter
        objects_text = f"Objects: {objects_detected}"
        cv2.putText(frame, objects_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detection classes
        if classes:
            classes_text = f"Classes: {', '.join(classes[:3])}"  # Show first 3 classes
            cv2.putText(frame, classes_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Status indicators
        status_y = height - 10
        
        # Session info
        session_info = f"Session: {os.path.basename(self.session_dir) if hasattr(self, 'session_dir') else 'N/A'}"
        cv2.putText(frame, session_info, (10, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Detection count
        detections_text = f"Detections: {len(self.detection_log)}"
        cv2.putText(frame, detections_text, (width - 200, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def stop_processing(self):
        """Stop processing and save final log"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        # Save final detection log
        self._save_detection_log()
        
        stats = {
            'total_frames': self.frame_count,
            'average_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'is_processing': False,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'avg_objects': np.mean(self.objects_history) if self.objects_history else 0,
            'total_detections': len(self.detection_log),
            'session_dir': self.session_dir if hasattr(self, 'session_dir') else None,
        }
        
        print(f"🛑 Processing stopped - {len(self.detection_log)} detections logged")
        return stats
    
    def _save_detection_log(self):
        """Save detection log to CSV file"""
        try:
            if not self.detection_log:
                return
            
            import csv
            import json
            
            csv_path = os.path.join(self.session_dir, "detection_log.csv")
            json_path = os.path.join(self.session_dir, "detection_log.json")
            
            # Save as CSV
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'frame_number', 'object_count', 'classes', 'fps']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for record in self.detection_log:
                    # Convert classes list to string
                    record_copy = record.copy()
                    record_copy['classes'] = ', '.join(record['classes']) if record['classes'] else ''
                    record_copy['timestamp'] = record['timestamp'].isoformat()
                    writer.writerow(record_copy)
            
            # Save as JSON
            with open(json_path, 'w') as jsonfile:
                json.dump(self.detection_log, jsonfile, default=str, indent=2)
            
            print(f"📊 Detection log saved: {csv_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving detection log: {e}")
    
    def get_current_frame(self):
        """Get current frame as JPEG"""
        if self.current_frame is None:
            return None
        
        _, jpeg = cv2.imencode('.jpg', self.current_frame)
        return jpeg.tobytes()
    
    def get_stats(self):
        """Get stats"""
        if not self.start_time:
            return {}
        
        current_fps = self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0
        
        return {
            'current_fps': current_fps,
            'total_frames': self.frame_count,
            'elapsed_time': time.time() - self.start_time,
            'is_processing': self.is_processing,
            'average_fps': np.mean(self.fps_history[-20:]) if self.fps_history else 0,
            'avg_objects': np.mean(self.objects_history[-20:]) if self.objects_history else 0,
            'total_detections': len(self.detection_log),
            'session_dir': self.session_dir if hasattr(self, 'session_dir') else None,
        }
    
    def get_detection_history(self):
        """Get recent detection history"""
        return self.detection_log[-10:]  # Last 10 detections

# Global instance
simple_video_processor = SimpleVideoProcessor()