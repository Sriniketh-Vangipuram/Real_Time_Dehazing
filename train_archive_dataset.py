import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow import keras

print("🚀 Starting REAL AI Training...")

def setup_archive_dataset():
    print("📁 Setting up archive dataset...")
    
    hazy_dir = "data/hazy"
    clear_dir = "data/clear"
    
    print(f"Using hazy: {hazy_dir}")
    print(f"Using clear: {clear_dir}")
    
    if not os.path.exists(hazy_dir):
        print(f"❌ Hazy folder not found!")
        return None, None
    
    if not os.path.exists(clear_dir):
        print(f"❌ Clear folder not found!")
        return None, None
    
    return hazy_dir, clear_dir

def load_dataset(hazy_dir, clear_dir, max_images=100, img_size=(256, 256)):
    """Load images from dataset"""
    print("📸 Loading images...")
    
    hazy_images = []
    clear_images = []
    
    # Get image files
    hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    clear_files = [f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Found {len(hazy_files)} hazy images and {len(clear_files)} clear images")
    
    # Use first max_images
    hazy_files = hazy_files[:max_images]
    clear_files = clear_files[:max_images]
    
    for i, (hazy_file, clear_file) in enumerate(zip(hazy_files, clear_files)):
        hazy_path = os.path.join(hazy_dir, hazy_file)
        clear_path = os.path.join(clear_dir, clear_file)
        
        # Load images
        hazy_img = cv2.imread(hazy_path)
        clear_img = cv2.imread(clear_path)
        
        if hazy_img is not None and clear_img is not None:
            # Resize
            hazy_img = cv2.resize(hazy_img, img_size)
            clear_img = cv2.resize(clear_img, img_size)
            
            # Normalize to [0, 1]
            hazy_img = hazy_img.astype(np.float32) / 255.0
            clear_img = clear_img.astype(np.float32) / 255.0
            
            hazy_images.append(hazy_img)
            clear_images.append(clear_img)
        
        if (i + 1) % 10 == 0:
            print(f"Loaded {i + 1}/{len(hazy_files)} image pairs")
    
    print(f"✅ Successfully loaded {len(hazy_images)} image pairs")
    return np.array(hazy_images), np.array(clear_images)

def train_model():
    """Train the CNN model"""
    print("🤖 Creating CNN model...")
    
    # Create model
    model = keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(256, 256, 3)),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2),
        
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        
        keras.layers.UpSampling2D(2),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
        keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✅ Model created and compiled!")
    
    # Load dataset
    hazy_dir, clear_dir = setup_archive_dataset()
    if hazy_dir is None:
        return
    
    X_train, y_train = load_dataset(hazy_dir, clear_dir, max_images=100)
    
    if len(X_train) == 0:
        print("❌ No images loaded!")
        return
    
    print(f"🎯 Starting training on {len(X_train)} images...")
    print("⏳ This will take 1-2 hours...")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=50,
        validation_split=0.2,
        verbose=1
    )
    
    # Save model
    os.makedirs("backend/media/models", exist_ok=True)
    model.save("backend/media/models/trained_model.h5")
    
    print("🎉 TRAINING COMPLETED!")
    print("📁 Model saved: backend/media/models/trained_model.h5")
    print("🚀 Restart Django server to use REAL AI!")

if __name__ == "__main__":
    train_model()