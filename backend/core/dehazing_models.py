import cv2
import numpy as np
import os
import tensorflow as tf


class WorkingDehazer:
    """
    Working dehazer that always returns visible images.
    Uses trained model if available, else applies a reliable enhancement.
    """

    def __init__(self):
        print("✅ Working Dehazer Initialized")
        self._model=None

    def dehaze(self, image_path, output_path=None):
        """Guaranteed working dehazing with fallbacks"""
        try:
            print(f"🔄 Processing: {os.path.basename(image_path)}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print("❌ Could not read image")
                return None, False

            print(f"📐 Original image size: {image.shape}")

            # Method 1: Try trained model
            enhanced, success = self.try_trained_model(image)
            if success:
                print("✅ Using trained model result")
            else:
                # Method 2: Fallback enhancement
                print("⚠️ Using reliable enhancement")
                enhanced = self.reliable_enhancement(image)

            # Save if needed
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, enhanced)
                print(f"💾 Image saved: {output_path}")

            print(f"✅ Final image size: {enhanced.shape}")
            return enhanced, True

        except Exception as e:
            print(f"❌ Dehazing error: {e}")
            import traceback
            traceback.print_exc()
            return None, False
        
    def _get_model(self):
        """Lazy load model - only when needed"""
        if self._model is None:
            model_path = "backend/media/models/trained_model.h5"
            # Alternative path for Render
            if not os.path.exists(model_path):
                model_path = "media/models/trained_model.h5"
            
            if os.path.exists(model_path):
                try:
                    self._model = tf.keras.models.load_model(model_path)
                    print("✅ Model loaded successfully")
                except:
                    print("⚠️ Model loading failed")
                    self._model = None
        return self._model
    # -----------------------------
    # 1️⃣ Try trained model first
    # -----------------------------
    def try_trained_model(self, image):
        """Use lazy-loaded model"""
        model = self._get_model()
        if model is None:
            return None, False
        """Try to use the trained CNN model"""
        try:
            model_path = "backend/media/models/trained_model.h5"

            if not os.path.exists(model_path):
                print("❌ No trained model found")
                return None, False

            model = tf.keras.models.load_model(model_path)

            original_size = image.shape[:2]
            processed = cv2.resize(image, (256, 256))
            processed = processed.astype(np.float32) / 255.0
            processed = np.expand_dims(processed, axis=0)

            # Predict
            enhanced = model.predict(processed, verbose=0)[0]
            enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
            enhanced = cv2.resize(enhanced, (original_size[1], original_size[0]))

            return enhanced, True

        except Exception as e:
            print(f"❌ Model prediction failed: {e}")
            return None, False

    # -----------------------------
    # 2️⃣ Fallback Reliable Enhancement
    # -----------------------------
    def reliable_enhancement(self, image):
        """Safe haze removal without block artifacts"""
        try:
            print("🌫️ Applying safe haze removal...")
        
        # Store original
            original = image.copy()
        
        # SIMPLE BUT EFFECTIVE approach - no complex math that can cause blocks
        # 1. Contrast enhancement in LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        
        # Strong CLAHE for haze reduction
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
        
        # 2. Merge back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Color saturation boost
            hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s_boosted = np.clip(s * 1.4, 0, 255).astype(np.uint8)  # 40% boost
            hsv_enhanced = cv2.merge([h, s_boosted, v])
            color_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # 4. Mild sharpening
            kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(color_enhanced, -1, kernel)
        
        # 5. Final blend (80% enhanced, 20% original)
            final = cv2.addWeighted(sharpened, 0.8, original, 0.2, 0)
        
            print("✅ Safe enhancement completed")
            return final
        
        except Exception as e:
            print(f"❌ Enhancement failed: {e}")
        # Return original if anything fails
        return image

    def get_dark_channel(self, image, window_size=15):
        """Calculate dark channel prior for haze estimation"""
        image_float = image.astype(np.float32)
        min_channel = np.min(image_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def estimate_atmospheric_light(self, image, dark_channel, top_percent=0.001):
        """Estimate atmospheric light from hazy image"""
        image_float = image.astype(np.float32)
        flat_dark = dark_channel.flatten()
        flat_image = image_float.reshape(-1, 3)

        num_pixels = max(1, int(flat_dark.size * top_percent))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        bright_pixels = flat_image[indices]

        atmospheric_light = np.max(bright_pixels, axis=0)
        return atmospheric_light

    def strong_contrast_enhancement(self, image):
        """Enhance contrast, saturation, and sharpness"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        l_min, l_max = np.percentile(l_enhanced, [1, 99])
        l_stretched = np.clip((l_enhanced - l_min) * 255.0 / (l_max - l_min), 0, 255).astype(np.uint8)

        lab_enhanced = cv2.merge([l_stretched, a, b])
        contrast_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s_boosted = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        hsv_enhanced = cv2.merge([h, s_boosted, v])
        saturated = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(saturated, -1, kernel)

        return sharpened


def get_dehazing_model():
    """Factory function to return instance"""
    return WorkingDehazer()