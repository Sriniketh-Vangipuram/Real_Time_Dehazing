import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def view_training_images():
    """View your training images side by side"""
    hazy_dir = "data/hazy"
    clear_dir = "data/clear"
    
    print("📸 Your Training Images:")
    print(f"Hazy images: {hazy_dir}")
    print(f"Clear images: {clear_dir}")
    
    # Get image files
    hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    clear_files = [f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(hazy_files)} hazy images")
    print(f"Found {len(clear_files)} clear images")
    
    if not hazy_files:
        print("❌ No hazy images found!")
        return
    
    # Show first 3 pairs
    for i in range(min(3, len(hazy_files))):
        hazy_path = os.path.join(hazy_dir, hazy_files[i])
        clear_path = os.path.join(clear_dir, clear_files[i]) if i < len(clear_files) else None
        
        print(f"\n--- Image Pair {i+1} ---")
        print(f"Hazy: {hazy_files[i]}")
        print(f"Clear: {clear_files[i] if clear_path else 'Not found'}")
        
        # Load and display images
        hazy_img = cv2.imread(hazy_path)
        if hazy_img is not None:
            print(f"Hazy image size: {hazy_img.shape}")
        
        if clear_path and os.path.exists(clear_path):
            clear_img = cv2.imread(clear_path)
            if clear_img is not None:
                print(f"Clear image size: {clear_img.shape}")
    
    # Save sample images to view in browser
    print(f"\n💡 To view images in browser:")
    print(f"1. Check folder: {os.path.abspath(hazy_dir)}")
    print(f"2. Check folder: {os.path.abspath(clear_dir)}")
    
    # Copy first image to media folder for web viewing
    if hazy_files:
        sample_hazy = os.path.join(hazy_dir, hazy_files[0])
        sample_clear = os.path.join(clear_dir, clear_files[0]) if clear_files else None
        
        # Copy to media folder for web access
        import shutil
        media_dir = "backend/media/training_samples"
        os.makedirs(media_dir, exist_ok=True)
        
        shutil.copy2(sample_hazy, os.path.join(media_dir, "sample_hazy.jpg"))
        if sample_clear:
            shutil.copy2(sample_clear, os.path.join(media_dir, "sample_clear.jpg"))
        
        print(f"✅ Sample images copied to: {media_dir}")
        print(f"🌐 View at: http://127.0.0.1:8000/media/training_samples/")

def check_image_sizes():
    """Check if all training images have proper sizes"""
    hazy_dir = "data/hazy"
    
    sizes = []
    for filename in os.listdir(hazy_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(hazy_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                sizes.append(img.shape)
    
    print(f"\n📊 Image size analysis:")
    print(f"Total images: {len(sizes)}")
    if sizes:
        unique_sizes = set(sizes)
        print(f"Unique sizes: {unique_sizes}")
        
        # Check for very small images that might cause blocks
        small_images = [size for size in sizes if size[0] < 100 or size[1] < 100]
        if small_images:
            print(f"⚠️ Warning: {len(small_images)} very small images found")
            print("Small images can cause blocky results when resized")

if __name__ == "__main__":
    view_training_images()
    check_image_sizes()