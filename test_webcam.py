# test_webcam.py (run this to test without Django)
import cv2
import time

def quick_webcam_test():
    """Simple test to verify webcam works"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam!")
        print("Possible solutions:")
        print("1. Make sure webcam is not being used by another app")
        print("2. Try camera index 1: cap = cv2.VideoCapture(1)")
        print("3. On Linux: sudo apt install v4l-utils")
        return False
    
    print("✅ Webcam opened successfully!")
    print("Press 'q' to quit...")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        current_fps = frame_count / (time.time() - start_time)
        
        # Display FPS
        cv2.putText(frame, f"Test FPS: {current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Test Results:")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {current_fps:.1f}")
    print("✅ Webcam test passed!" if frame_count > 0 else "❌ Webcam test failed")
    
    return frame_count > 0

if __name__ == "__main__":
    quick_webcam_test()