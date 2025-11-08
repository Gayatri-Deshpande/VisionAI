# predict.py
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def load_model(model_path):
    """Load trained model"""
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def load_class_names(json_path):
    """Load class indices and invert dictionary"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Class indices file not found at {json_path}")
    with open(json_path, "r") as f:
        class_indices = json.load(f)
    # invert key-value to map index -> label
    class_names = {v: k for k, v in class_indices.items()}
    # sort to ensure correct order
    return [class_names[i] for i in sorted(class_names.keys())]

def predict_image(model, img_path, class_names):
    """Predict DR class for one image"""
    x = preprocess_image(img_path)
    preds = model.predict(x)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)
    return class_names[class_id], confidence

def combine_predictions(left_pred, right_pred):
    """Combine both eye predictions intelligently"""
    # Convert labels to severity order (No_DR=0 → Proliferative_DR=4)
    severity_order = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferative_DR": 4
    }

    left_severity = severity_order.get(left_pred[0], 0)
    right_severity = severity_order.get(right_pred[0], 0)

    # Take the more severe result
    final_severity = max(left_severity, right_severity)
    final_label = [k for k, v in severity_order.items() if v == final_severity][0]
    avg_conf = (left_pred[1] + right_pred[1]) / 2
    return final_label, avg_conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DR stage from fundus images")
    parser.add_argument("--model", type=str, default=r"E:\DRdetection\models\best_model.h5", help="Path to trained model")
    parser.add_argument("--class_map", type=str, default=r"E:\DRdetection\models\class_indices.json", help="Path to class index map")
    parser.add_argument("--left", type=str, help="Path to left eye image")
    parser.add_argument("--right", type=str, help="Path to right eye image")
    parser.add_argument("--image", type=str, help="Single image path (if not using left/right)")
    args = parser.parse_args()

    # Load model and classes
    model = load_model(args.model)
    class_names = load_class_names(args.class_map)
    print("✅ Model and class labels loaded successfully.")

    # Single image prediction
    if args.image:
        pred_label, conf = predict_image(model, args.image, class_names)
        print(f"Prediction for {args.image}: {pred_label} (Confidence: {conf:.2f})")

    # Two-eye combined prediction
    elif args.left and args.right:
        left_pred = predict_image(model, args.left, class_names)
        right_pred = predict_image(model, args.right, class_names)
        combined_label, avg_conf = combine_predictions(left_pred, right_pred)
        print(f"Left Eye:  {left_pred[0]} ({left_pred[1]:.2f})")
        print(f"Right Eye: {right_pred[0]} ({right_pred[1]:.2f})")
        print(f"➡️ Combined Final Diagnosis: {combined_label} (Avg Confidence: {avg_conf:.2f})")
    else:
        print("⚠️ Please provide either --image OR both --left and --right arguments.")
