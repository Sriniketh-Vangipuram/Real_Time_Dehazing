import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow import keras

def create_simple_dataset():
    """
    Create a simple dataset from your 100 images
    Place hazy images in 'data/hazy/' and clear in 'data/clear/'
    """
    print("📁 Preparing 100 image dataset...")
    
    # Your 100 image pairs go here:
    # data/hazy/image1.jpg, image2.jpg...
    # data/clear/image1.jpg, image2.jpg...
    
    hazy_dir = "data/hazy"
    clear_dir = "data/clear"
    
    # Check if dataset exists
    if not os.path.exists(hazy_dir):
        print("⚠️ Please create 'data/hazy/' and 'data/clear/' folders with your 100 images")
        return None, None
    
    return hazy_dir, clear_dir

def load_and_preprocess_data(hazy_dir, clear_dir, img_size=(256, 256)):
    """Load and preprocess your 100 images"""
    hazy_images = []
    clear_images = []
    
    # Get image files
    hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    clear_files = [f for f in os.listdir(clear_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"📊 Found {len(hazy_files)} hazy and {len(clear_files)} clear images")
    
    # Load images (first 100 or available)
    for i, (hazy_file, clear_file) in enumerate(zip(hazy_files[:100], clear_files[:100])):
        hazy_path = os.path.join(hazy_dir, hazy_file)
        clear_path = os.path.join(clear_dir, clear_file)
        
        # Load and resize
        hazy_img = cv2.imread(hazy_path)
        clear_img = cv2.imread(clear_path)
        
        if hazy_img is not None and clear_img is not None:
            hazy_img = cv2.resize(hazy_img, img_size)
            clear_img = cv2.resize(clear_img, img_size)
            
            # Normalize to [0, 1]
            hazy_img = hazy_img.astype(np.float32) / 255.0
            clear_img = clear_img.astype(np.float32) / 255.0
            
            hazy_images.append(hazy_img)
            clear_images.append(clear_img)
    
    return np.array(hazy_images), np.array(clear_images)

def train_model():
    """Train the model on your 100 images"""
    print("🚀 Starting training on 100 images...")
    
    # Create dataset
    hazy_dir, clear_dir = create_simple_dataset()
    if hazy_dir is None:
        return
    
    # Load data
    X_train, y_train = load_and_preprocess_data(hazy_dir, clear_dir)
    
    if len(X_train) == 0:
        print("❌ No images found! Please add images to data folders")
        return
    
    print(f"🎯 Training on {len(X_train)} image pairs")
    
    # Create model (same architecture as in dehazing_models.py)
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
    
    print("🤖 Model architecture created")
    print("⏳ Training started... (This may take 1-2 hours)")
    
    # Train for 50 epochs (adjust based on your time)
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
    
    print("✅ Training completed!")
    print("📁 Model saved: backend/media/models/trained_model.h5")
    print("🎉 Now restart Django server to use REAL AI dehazing!")

if __name__ == "__main__":
    train_model()