import os
import shutil
import cv2

def test_with_dataset_images():
    """Copy dataset images to media folder for testing"""
    
    # Paths to your dataset
    dataset_hazy = "data/hazy"
    dataset_clear = "data/clear"
    
    # CORRECT PATH - relative to project root
    test_dir = "backend/media/test_dataset"
    os.makedirs(test_dir, exist_ok=True)
    
    print("📁 Testing with dataset images...")
    
    # Get all image files from dataset
    hazy_files = [f for f in os.listdir(dataset_hazy) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(hazy_files)} hazy images")
    
    # Filter for larger images (better for object detection)
    large_images = []
    
    for hazy_file in hazy_files[:10]:  # Test first 10 images
        hazy_path = os.path.join(dataset_hazy, hazy_file)
        
        # Check image size
        img = cv2.imread(hazy_path)
        if img is not None:
            height, width = img.shape[:2]
            if height >= 300 and width >= 300:  # Only use large images
                large_images.append((hazy_file, height, width))
    
    print(f"📏 Found {len(large_images)} large images suitable for testing")
    
    # Copy test images to media folder
    test_count = min(5, len(large_images))  # Copy max 5 images
    copied_images = []
    
    for i, (hazy_file, height, width) in enumerate(large_images[:test_count]):
        # Copy hazy image to test directory
        source_path = os.path.join(dataset_hazy, hazy_file)
        dest_path = os.path.join(test_dir, f"test_{i+1}_{hazy_file}")
        shutil.copy2(source_path, dest_path)
        copied_images.append((f"test_{i+1}_{hazy_file}", height, width))
        print(f"✅ Copied: {hazy_file} ({width}×{height}px) → {dest_path}")
    
    print(f"\n🎯 Ready to test! Upload these images from: {os.path.abspath(test_dir)}")
    print("📋 Test images copied:")
    for img_name, height, width in copied_images:
        print(f"   - {img_name} ({width}×{height}px)")
    
    print(f"\n💡 To test in browser:")
    print(f"1. Start server: cd backend && python manage.py runserver")
    print(f"2. Go to: http://127.0.0.1:8000/browse-dataset/")
    print(f"3. Click 'Test This Image' on any image")
if __name__ == "__main__":
    test_with_dataset_images()