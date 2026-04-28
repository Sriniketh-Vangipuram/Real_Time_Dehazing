import cv2
import numpy as np
import time
import os
import threading
from datetime import datetime
import tempfile

# Try importing models
try:
    from .dehazing_models import get_dehazing_model
    from .object_detection import get_object_detector
    MODELS_AVAILABLE = True
except Exception as e:
    print(f"❌ Model import error: {e}")
    MODELS_AVAILABLE = False


class VideoProcessor:
    def __init__(self):
        self.is_processing = False
        self.current_frame = None
        self.processing_thread = None

        self.frame_count = 0
        self.start_time = None
        self.fps_history = []

        self.dehazing_times = []
        self.detection_times = []
        self.objects_detected_history = []

        self.dehazer = None
        self.detector = None
        self._initialize_models()

        self.last_detection_frame = None
        self.last_detection_count = 0

    # --------------------------------------------------
    # MODEL INITIALIZATION
    # --------------------------------------------------
    def _initialize_models(self):
        print("🔄 Initializing models...")

        if not MODELS_AVAILABLE:
            print("⚠️ Running in SIMULATION mode")
            return

        try:
            self.dehazer = get_dehazing_model()
            print("✅ Dehazing model loaded")
        except Exception as e:
            print(f"❌ Dehazing model failed: {e}")
            self.dehazer = None

        try:
            self.detector = get_object_detector()
            print("✅ Object detector loaded")
        except Exception as e:
            print(f"❌ Object detector failed: {e}")
            self.detector = None

    # --------------------------------------------------
    # START / STOP
    # --------------------------------------------------
    def start_processing(self, camera_index=0):
        if self.is_processing:
            return False

        self.is_processing = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_history.clear()
        self.dehazing_times.clear()
        self.detection_times.clear()
        self.objects_detected_history.clear()

        self.processing_thread = threading.Thread(
            target=self._process_video,
            args=(camera_index,),
            daemon=True
        )
        self.processing_thread.start()

        print(f"🎬 Video started @ {datetime.now()}")
        return True

    def stop_processing(self):
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        print("🛑 Video stopped")

    # --------------------------------------------------
    # VIDEO LOOP
    # --------------------------------------------------
    def _process_video(self, camera_index):
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("❌ Cannot open camera")
            self.is_processing = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("📹 Camera opened")

        while self.is_processing:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)

            frame = self._process_frame(frame, fps)
            self.current_frame = frame

            time.sleep(0.005)

        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released")

    # --------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------
    def _process_frame(self, frame, fps):
        try:
            processed = frame.copy()
            h, w = frame.shape[:2]
            objects_detected = 0

            # ---------- DEHAZING ----------
            if self.dehazer and self.frame_count % 10 == 0:
                start = time.time()
                try:
                    temp_path = os.path.join(
                        tempfile.gettempdir(),
                        f"frame_{self.frame_count}.jpg"
                    )

                    cv2.imwrite(temp_path, frame)
                    dehazed, success = self.dehazer.dehaze(temp_path, None)

                    if success and dehazed is not None:
                        if dehazed.shape[:2] != (h, w):
                            dehazed = cv2.resize(dehazed, (w, h))
                        processed = cv2.addWeighted(frame, 0.3, dehazed, 0.7, 0)

                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                    self.dehazing_times.append(time.time() - start)
                except Exception as e:
                    print(f"⚠️ Dehazing error: {e}")

            # ---------- OBJECT DETECTION ----------
            if self.detector and self.frame_count % 3 == 0:
                start = time.time()
                try:
                    objects_detected, detected = self._detect_objects_with_persistence(
                        processed, 0.25
                    )
                    processed = detected
                    self.objects_detected_history.append(objects_detected)
                    self.detection_times.append(time.time() - start)
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")

            # ---------- OVERLAYS ----------
            self._draw_overlays(processed, fps, objects_detected)

            return processed

        except Exception as e:
            print(f"❌ Frame error: {e}")
            return frame

    # --------------------------------------------------
    # OBJECT DETECTION WITH PERSISTENCE
    # --------------------------------------------------
    def _detect_objects_with_persistence(self, frame, threshold):
        count, detected = self.detector.detect_objects_simple(frame, threshold)

        if detected is not None and count > 0:
            self.last_detection_frame = detected
            self.last_detection_count = count
            return count, detected

        if self.last_detection_frame is not None:
            return self.last_detection_count, self.last_detection_frame

        return 0, frame

    # --------------------------------------------------
    # OVERLAYS
    # --------------------------------------------------
    def _draw_overlays(self, frame, fps, objects):
        h, w = frame.shape[:2]

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Objects: {objects}", (w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        mode = "REAL AI" if (self.dehazer or self.detector) else "SIMULATION"
        cv2.putText(frame, f"Mode: {mode}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 200, 0), 2)

    # --------------------------------------------------
    # STREAMING + STATS
    # --------------------------------------------------
    def get_current_frame(self):
        if self.current_frame is None:
            return None
        _, jpeg = cv2.imencode(".jpg", self.current_frame)
        return jpeg.tobytes()

    def get_stats(self):
        if not self.start_time:
            return {}

        return {
            "fps": np.mean(self.fps_history[-30:]) if self.fps_history else 0,
            "frames": self.frame_count,
            "dehazer": self.dehazer is not None,
            "detector": self.detector is not None,
            "avg_dehaze_ms": np.mean(self.dehazing_times[-20:]) * 1000 if self.dehazing_times else 0,
            "avg_detect_ms": np.mean(self.detection_times[-20:]) * 1000 if self.detection_times else 0,
        }


# --------------------------------------------------
# GLOBAL INSTANCE (IMPORTANT)
# --------------------------------------------------
video_processor = VideoProcessor()
