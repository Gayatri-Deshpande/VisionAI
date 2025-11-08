#save_class_indices.py
import os
import json
import pandas as pd
from data_generator import get_generators
from utils import read_csv

# ========================
# PATHS (same as in train.py)
# ========================
BASE_DIR   = r"E:\DRdetection\dataset"
TRAIN_CSV  = os.path.join(BASE_DIR, "train_1.csv")
VALID_CSV  = os.path.join(BASE_DIR, "valid.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test.csv")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
VAL_IMG_DIR   = os.path.join(BASE_DIR, "val_images")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "test_images")

MODEL_DIR  = r"E:\DRdetection\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================
# Load CSVs
# ========================
train_df = read_csv(TRAIN_CSV)
valid_df = read_csv(VALID_CSV)
test_df  = read_csv(TEST_CSV)

# Convert labels to strings (required)
train_df['diagnosis'] = train_df['diagnosis'].astype(str)
valid_df['diagnosis'] = valid_df['diagnosis'].astype(str)
test_df['diagnosis']  = test_df['diagnosis'].astype(str)

# ========================
# Create data generators
# ========================
train_gen, val_gen, test_gen = get_generators(
    train_df, valid_df, test_df,
    train_dir=TRAIN_IMG_DIR,
    val_dir=VAL_IMG_DIR,
    test_dir=TEST_IMG_DIR,
    batch_size=8  # small batch, not training
)

# ========================
# Save class indices
# ========================
json_path = os.path.join(MODEL_DIR, "class_indices.json")
with open(json_path, "w") as f:
    json.dump(train_gen.class_indices, f)

print(f"✅ class_indices.json saved successfully at: {json_path}")
