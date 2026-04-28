from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.conf import settings
from .models import ProcessingResult
from .utils import process_single_image, detect_haze
import os
import time
import shutil
import cv2
import uuid


def home(request):
    return render(request, 'core/home.html')


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            uploaded_file = request.FILES['image']
            result = ProcessingResult(original_image=uploaded_file)
            result.save()
            return redirect('process_image', result_id=result.id)
        except Exception as e:
            return render(request, 'core/upload.html', {
                'error': f'Error uploading image: {str(e)}'
            })

    return render(request, 'core/upload.html')


def process_image(request, result_id):
    result = get_object_or_404(ProcessingResult, id=result_id)

    try:
        start_time = time.time()

        print(f"🚀 Processing started for ID: {result_id}")

        # STEP 1: Haze Detection
        is_hazy, confidence = detect_haze(result.original_image.path)

        # STEP 2: Paths
        output_filename = f"dehazed_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(settings.MEDIA_ROOT, 'output', output_filename)

        detection_orig_filename = f"detection_orig_{uuid.uuid4().hex[:8]}.jpg"
        detection_enh_filename = f"detection_enh_{uuid.uuid4().hex[:8]}.jpg"

        detection_orig_path = os.path.join(settings.MEDIA_ROOT, 'output', detection_orig_filename)
        detection_enh_path = os.path.join(settings.MEDIA_ROOT, 'output', detection_enh_filename)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # STEP 3: Dehazing + Detection
        processing_result = process_single_image(
            result.original_image.path,
            output_path
        )

        if not processing_result['success']:
            return render(request, 'core/process.html', {
                'result': result,
                'error': processing_result.get('error', 'Processing failed')
            })

        # STEP 4: Save Base Results
        result.dehazed_image.name = f'output/{output_filename}'
        result.is_hazy = is_hazy
        result.haze_confidence = confidence
        result.processing_time = processing_result['processing_time']

        # Metrics
        if 'metrics' in processing_result:
            result.enhanced_psnr = processing_result['metrics']['psnr']
            result.enhanced_ssim = processing_result['metrics']['ssim']
        else:
            result.enhanced_psnr = 0.0
            result.enhanced_ssim = 0.0

        # STEP 5: Object Detection Results
        if 'object_counts' in processing_result:
            obj = processing_result['object_counts']

            result.objects_detected_before = obj['original']
            result.objects_detected_after = obj['enhanced']

            # ✅ CORRECT IMPROVEMENT LOGIC
            if obj['original'] > 0:
                result.detection_improvement = (
                    (obj['enhanced'] - obj['original']) / obj['original']
                ) * 100
            else:
                result.detection_improvement = 100.0 if obj['enhanced'] > 0 else 0.0

        else:
            result.objects_detected_before = 0
            result.objects_detected_after = 0
            result.detection_improvement = 0.0

        # STEP 6: Save Detection Images
        if 'detection_results' in processing_result:
            det = processing_result['detection_results']

            if 'original_detection' in det and os.path.exists(det['original_detection']):
                shutil.copy2(det['original_detection'], detection_orig_path)
                result.detection_original.name = f'output/{detection_orig_filename}'

            if 'enhanced_detection' in det and os.path.exists(det['enhanced_detection']):
                shutil.copy2(det['enhanced_detection'], detection_enh_path)
                result.detection_enhanced.name = f'output/{detection_enh_filename}'

        # ❌ REMOVED BUG: DO NOT CALL calculate_improvement()

        result.save()

        total_time = time.time() - start_time

        # STEP 7: Status Logic
        if result.objects_detected_before == 0 and result.objects_detected_after > 0:
            status = "🚀 MAJOR IMPROVEMENT"
        elif result.detection_improvement > 0:
            status = "✅ IMPROVED"
        elif result.detection_improvement == 0:
            status = "⚠️ NO CHANGE"
        else:
            status = "❌ WORSE"

        print(f"✅ Done in {total_time:.2f}s | Status: {status}")

        return redirect('view_results', result_id=result.id)

    except Exception as e:
        print(f"❌ ERROR: {e}")

        result.processing_time = time.time() - start_time
        result.is_hazy = False
        result.haze_confidence = 0.0
        result.objects_detected_before = 0
        result.objects_detected_after = 0
        result.detection_improvement = 0.0
        result.save()

        return render(request, 'core/process.html', {
            'result': result,
            'error': str(e)
        })


def view_results(request, result_id):
    result = get_object_or_404(ProcessingResult, id=result_id)

    return render(request, 'core/results.html', {
        'result': result
    })


# ================= VIDEO =================

from .simple_video_processor import simple_video_processor as video_processor


def video_processing(request):
    has_webcam = False
    try:
        cap = cv2.VideoCapture(0)
        has_webcam = cap.isOpened()
        cap.release()
    except:
        pass

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'start':
            success = video_processor.start_processing()
            return JsonResponse({'status': 'started' if success else 'error'})

        elif action == 'stop':
            stats = video_processor.stop_processing()
            return JsonResponse({'status': 'stopped', 'stats': stats})

        elif action == 'stats':
            return JsonResponse(video_processor.get_stats())

        elif action == 'check_webcam':
            return JsonResponse({'has_webcam': has_webcam})

    return render(request, 'core/video.html', {'has_webcam': has_webcam})


def video_feed(request):
    def generate_frames():
        import numpy as np
        import time

        while True:
            frame = video_processor.get_current_frame()

            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Starting...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', blank)

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            time.sleep(0.03)

    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def browse_dataset(request):
    """Browse and test dataset images"""
    import glob
    
    test_dir = "backend/media/test_dataset"
    dataset_images = []
    
    if os.path.exists(test_dir):
        image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            # Get image dimensions
            img = cv2.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                dataset_images.append({
                    'name': img_name,
                    'path': f"/media/test_dataset/{img_name}",
                    'width': width,
                    'height': height,
                    'size': f"{width}×{height}"
                })
    
    context = {
        'dataset_images': dataset_images,
    }
    return render(request, 'core/browse_dataset.html', context)
# Add new view for detection history
def detection_history(request):
    """Show detection history from video processing"""
    from .simple_video_processor import simple_video_processor
    
    # Get detection history
    detection_history = simple_video_processor.get_detection_history()
    stats = simple_video_processor.get_stats()
    
    # List saved sessions
    sessions_dir = "media/detection_logs"
    sessions = []
    
    if os.path.exists(sessions_dir):
        for session in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, session)
            if os.path.isdir(session_path):
                # Count images in session
                images = [f for f in os.listdir(session_path) if f.endswith('.jpg')]
                sessions.append({
                    'name': session,
                    'path': session_path,
                    'image_count': len(images),
                    'created': os.path.getctime(session_path)
                })
    
    context = {
        'detection_history': detection_history,
        'stats': stats,
        'sessions': sorted(sessions, key=lambda x: x['created'], reverse=True)[:5],  # Last 5 sessions
        'has_sessions': len(sessions) > 0,
    }
    
    return render(request, 'core/detection_history.html', context)