#predict_pair.py
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import json

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

def load_model_fn(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path, target_size=(224,224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

def predict_single(model, img_path):
    x = preprocess_image(img_path)
    preds = model.predict(x)[0]
    return preds

def combine_probs(p1, p2, mode='max'):
    if p1 is None: return p2
    if p2 is None: return p1
    if mode == 'avg':
        return (p1 + p2) / 2.0
    return np.maximum(p1, p2)  # default max

def pretty_print_probs(probs):
    for i, p in enumerate(probs):
        print(f"  {i} - {CLASS_NAMES[i]} : {p:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=str, default=None)
    parser.add_argument("--right", type=str, default=None)
    parser.add_argument("--model", type=str, default=os.path.join("models", "best_model.h5"))
    parser.add_argument("--mode", type=str, default="max", choices=["max","avg"])
    args = parser.parse_args()

    model = load_model_fn(args.model)
    p_left = predict_single(model, args.left) if args.left else None
    p_right = predict_single(model, args.right) if args.right else None

    print("Left eye prediction:")
    pretty_print_probs(p_left) if p_left is not None else print("  No left image provided.")
    print("\nRight eye prediction:")
    pretty_print_probs(p_right) if p_right is not None else print("  No right image provided.")

    combined = combine_probs(p_left, p_right, mode=args.mode)
    cid = int(np.argmax(combined))
    print("\nCombined prediction:", cid, CLASS_NAMES[cid])
