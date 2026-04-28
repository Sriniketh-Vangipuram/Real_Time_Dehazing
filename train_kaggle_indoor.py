import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow import keras
import zipfile

def setup_kaggle_dataset():
    """Setup Kaggle Indoor Training Set (ITS)"""
    print("📁 Setting up Kaggle Indoor dataset...")
    
    # If you have indoor.zip, extract it
    if os.path.exists("indoor.zip") and not os.path.exists("data/indoor"):
        print("📦 Extracting indoor.zip...")
        with zipfile.ZipFile("indoor.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")
    
    indoor_path = "data/indoor"
    if not os.path.exists(indoor_path):
        print("❌ indoor folder not found! Please extract indoor.zip to data/")
        return None, None
    
    hazy_dir = os.path.join(indoor_path, "hazy")
    clear_dir = os.path.join(indoor_path, "clear")
    
    if not os.path.exists(hazy_dir):
        print("❌ hazy folder not found in indoor/")
        return None, None
    
    print(f"✅ Found Kaggle Indoor dataset at {indoor_path}")
    return hazy_dir, clear_dir

def load_kaggle_data(hazy_dir, clear_dir, max_images=100, img_size=(256, 256)):
    """Load images from Kaggle Indoor dataset"""
    hazy_images = []
    clear_images = []
    
    # Get hazy image files
    hazy_files = [f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"📊 Found {len(hazy_files)} hazy images in dataset")
    
    # Use first max_images (or all if less)
    hazy_files = hazy_files[:max_images]
    
    for i, hazy_file in enumerate(hazy_files):
        hazy_path = os.path.join(hazy_dir, hazy_file)
        
        # Find corresponding clear image
        # Kaggle format: 'hazy_image.png' -> 'clear_image.png'
        clear_file = hazy_file  # Usually same filename in clear folder
        clear_path = os.path.join(clear_dir, clear_file)
        
        # If clear folder doesn't exist or file not found, try alternative names
        if not os.path.exists(clear_path):
            # Try common naming patterns
            clear_file = hazy_file.replace('hazy', 'clear')
            clear_path = os.path.join(clear_dir, clear_file)
        
        if not os.path.exists(clear_path):
            clear_file = hazy_file.replace('_hazy', '_clear')
            clear_path = os.path.join(clear_dir, clear_file)
        
        # Load images
        hazy_img = cv2.imread(hazy_path)
        clear_img = cv2.imread(clear_path) if os.path.exists(clear_path) else None
        
        if hazy_img is not None and clear_img is not None:
            # Resize
            hazy_img = cv2.resize(hazy_img, img_size)
            clear_img = cv2.resize(clear_img, img_size)
            
            # Normalize
            hazy_img = hazy_img.astype(np.float32) / 255.0
            clear_img = clear_img.astype(np.float32) / 255.0
            
            hazy_images.append(hazy_img)
            clear_images.append(clear_img)
            
            if (i + 1) % 10 == 0:
                print(f"📸 Loaded {i + 1}/{len(hazy_files)} image pairs")
    
    print(f"✅ Successfully loaded {len(hazy_images)} image pairs")
    return np.array(hazy_images), np.array(clear_images)

def train_on_kaggle():
    """Train model on Kaggle Indoor dataset"""
    print("🚀 Training on Kaggle Indoor Dataset...")
    
    # Setup dataset
    hazy_dir, clear_dir = setup_kaggle_dataset()
    if hazy_dir is None:
        return
    
    # Load data (use 100 images or all available)
    X_train, y_train = load_kaggle_data(hazy_dir, clear_dir, max_images=100)
    
    if len(X_train) == 0:
        print("❌ No valid image pairs found!")
        return
    
    print(f"🎯 Training on {len(X_train)} image pairs from Kaggle")
    
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
    
    print("🤖 Model architecture created")
    print("⏳ Training started... (1-2 hours)")
    
    # Train
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
    
    print("✅ Training completed on Kaggle dataset!")
    print("📁 Model saved: backend/media/models/trained_model.h5")
    print("🎉 Restart Django server to use your trained model!")

if __name__ == "__main__":
    train_on_kaggle()