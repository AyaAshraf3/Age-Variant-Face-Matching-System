# ==============================================================================
#  Age-Variant Face Verification System with Keras 3 Compatibility
# ==============================================================================
#
#  This script provides a stable and conflict-free solution by using a modern,
#  fully Keras 3-compatible pipeline:
#
#  1. Keras 3 / TensorFlow: To load and run the custom age estimation model.
#  2. MediaPipe: A lightweight and modern library for high-performance face
#     detection.
#  3. Keras-FaceNet: A Keras-native implementation of the powerful FaceNet model
#     for face verification, ensuring no dependency conflicts.
#
# ==============================================================================

# --- Imports ---
import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import keras
import mediapipe as mp
from keras_facenet import FaceNet
import matplotlib.pyplot as plt

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_age_model(model_path):
    """
    Loads the custom .keras age estimation model in a Keras 3 environment.

    Args:
        model_path (str): Path to the .keras model file.

    Returns:
        keras.Model: The loaded Keras model, or None on failure.
    """
    if not (model_path and os.path.exists(model_path)):
        print(f"Error: Model file not found at '{model_path}'.")
        return None
    try:
        print(f"Loading custom age model from: {model_path}")
        model = keras.models.load_model(model_path, compile=False)
        print("Custom age model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the Keras model: {e}")
        return None

def detect_and_crop_face_mediapipe(image_path):
    """
    Detects a face in an image using MediaPipe and returns a cropped version.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: The cropped face in RGB format, or None on failure.
    """
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if not results.detections:
        raise ValueError(f"No face detected in {os.path.basename(image_path)}.")

    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = img.shape
    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

    padding = 20
    x, y = max(0, x - padding), max(0, y - padding)
    w, h = w + (padding * 2), h + (padding * 2)

    face_crop_rgb = img_rgb[y:y+h, x:x+w]
    return face_crop_rgb

def preprocess_face_for_age_model(face_crop_rgb, target_size=(224, 224)):
    """
    Preprocesses a cropped RGB face image for the age estimation model.
    """
    resized_face = cv2.resize(face_crop_rgb, (target_size[1], target_size[0]))
    preprocessed_face = tf.keras.applications.resnet50.preprocess_input(resized_face)
    return np.expand_dims(preprocessed_face, axis=0)

def display_results(img_a_path, img_b_path, age_a, age_b, is_match, distance, threshold):
    """
    Displays the input images, estimated ages, and final verification result.
    """
    img_a = cv2.imread(img_a_path)
    img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
    img_b = cv2.imread(img_b_path)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_a)
    axes[0].set_title(f"Image A\nEstimated Age: {age_a:.1f} years", fontsize=14)
    axes[0].axis('off')
    axes[1].imshow(img_b)
    axes[1].set_title(f"Image B\nEstimated Age: {age_b:.1f} years", fontsize=14)
    axes[1].axis('off')

    if is_match:
        title_text, title_color = "Result: Same Person", 'green'
    else:
        title_text, title_color = "Result: Different People", 'red'

    plt.suptitle(title_text, fontsize=16, color=title_color, y=1.02)
    plt.tight_layout()

    print("\n" + "="*50)
    print("          Face Verification Results")
    print("="*50)
    print(f"VERIFIED: {is_match}")
    print(f"\n- Analysis Details:")
    print(f"  - Verification Library: Keras-FaceNet")
    print(f"  - Cosine Distance: {distance:.4f}")
    print(f"  - Dynamic Similarity Threshold: {threshold:.4f}")
    print(f"    (A distance score < {threshold} means a match)")
    print("="*50)
    plt.show()

def main(args):
    """
    Main execution function for the age-invariant verification pipeline.
    """
    age_model = load_age_model(args.age_model)
    facenet_embedder = FaceNet()

    if not age_model:
        return

    try:
        # --- Step 1: Detect and Crop Faces using MediaPipe ---
        print("\n Detecting faces with MediaPipe...")
        face_a_crop = detect_and_crop_face_mediapipe(args.image1)
        face_b_crop = detect_and_crop_face_mediapipe(args.image2)
        print("Face detection complete.")

        # --- Step 2: Estimate Age for each face ---
        print("\n Estimating ages with custom model...")
        preprocessed_face_a = preprocess_face_for_age_model(face_a_crop)
        estimated_age_a = age_model.predict(preprocessed_face_a, verbose=0)[0][0]
        print(f"   - Estimated Age for Image A: {estimated_age_a:.1f}")

        preprocessed_face_b = preprocess_face_for_age_model(face_b_crop)
        estimated_age_b = age_model.predict(preprocessed_face_b, verbose=0)[0][0]
        print(f"   - Estimated Age for Image B: {estimated_age_b:.1f}")

        # --- Step 3: Perform Face Verification using Keras-FaceNet ---
        print("\n Performing age-invariant face verification...")
        face_a_batch = np.expand_dims(face_a_crop, axis=0)
        face_b_batch = np.expand_dims(face_b_crop, axis=0)
        embedding_a = facenet_embedder.embeddings(face_a_batch)[0]
        embedding_b = facenet_embedder.embeddings(face_b_batch)[0]
        from scipy.spatial.distance import cosine
        distance = cosine(embedding_a, embedding_b)
        print("Verification complete.")

        # --- Step 4: Dynamic Threshold Logic ---
        age_difference = abs(estimated_age_a - estimated_age_b)
        dynamic_threshold = 0.40  # Standard threshold for FaceNet

        if age_difference > 40:
            dynamic_threshold = 0.65
        elif age_difference > 25:
            dynamic_threshold = 0.50

        is_match = distance < dynamic_threshold
        print(f"   - Age difference is {age_difference:.1f} years. Using dynamic threshold of {dynamic_threshold:.2f}.")

        # --- Step 5: Display Final Results ---
        display_results(
            img_a_path=args.image1, img_b_path=args.image2,
            age_a=estimated_age_a, age_b=estimated_age_b,
            is_match=is_match, distance=distance, threshold=dynamic_threshold)

    except ValueError as ve:
         print(f"\n An error occurred. Details: {ve}")
    except Exception as e:
        print(f"\n An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age-Invariant Face Verification System")
    parser.add_argument("--image1", type=str, required=True, help="Path to the first image.")
    parser.add_argument("--image2", type=str, required=True, help="Path to the second image.")
    parser.add_argument("--age_model", type=str, required=True, help="Path to the custom .keras age model file.")
    
    args = parser.parse_args()
    main(args)