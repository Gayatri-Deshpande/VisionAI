# train.py
import os
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger

from data_generator import get_generators
from model import build_vgg16_model
from utils import read_csv, get_class_weights




# ========================
# PATHS
# ========================
BASE_DIR   = r"E:\DRdetection\dataset"
TRAIN_CSV  = os.path.join(BASE_DIR, "train_1.csv")
VALID_CSV  = os.path.join(BASE_DIR, "valid.csv")
TEST_CSV   = os.path.join(BASE_DIR, "test.csv")

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
VAL_IMG_DIR   = os.path.join(BASE_DIR, "val_images")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "test_images")

MODEL_DIR  = r"E:\DRdetection\models"
OUTPUT_DIR = r"E:\DRdetection\outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# CONFIG
# ========================
BATCH_SIZE = 16
EPOCHS = 30
INPUT_SHAPE = (224,224,3)

# ========================
# Load CSVs
# ========================
train_df = read_csv(TRAIN_CSV)
valid_df = read_csv(VALID_CSV)
test_df  = read_csv(TEST_CSV)

# Convert labels to strings (redundant if already done in data_generator.py)
train_df['diagnosis'] = train_df['diagnosis'].astype(str)
valid_df['diagnosis'] = valid_df['diagnosis'].astype(str)
test_df['diagnosis']  = test_df['diagnosis'].astype(str)

# ========================
# Generators
# ========================
train_gen, val_gen, test_gen = get_generators(
    train_df, valid_df, test_df,
    train_dir=TRAIN_IMG_DIR,
    val_dir=VAL_IMG_DIR,
    test_dir=TEST_IMG_DIR,
    batch_size=BATCH_SIZE
)
with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
    json.dump(train_gen.class_indices, f)
print("✅ Saved class_indices.json at:", os.path.join(MODEL_DIR, 'class_indices.json'))


# ========================
# Class weights
# ========================
class_weight = get_class_weights(train_df)
print('Class weights:', class_weight)

# ========================
# Build model
# ========================
model = build_vgg16_model(
    input_shape=INPUT_SHAPE, 
    n_classes=5, 
    dropout=0.5
)
model.compile(
    optimizer=Adam(learning_rate=1e-4), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)
model.summary()

# ========================
# Callbacks
# ========================
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_model.h5'),
    monitor='val_loss', 
    save_best_only=True, 
    verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
csv_logger = CSVLogger(os.path.join(OUTPUT_DIR, 'training_log.csv'))

# ========================
# Stage 1: Train top layers
# ========================
print("Training top layers...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[checkpoint, reduce_lr, early, csv_logger]
)

# ========================
# Stage 2: Fine-tuning
# ========================
print("Fine-tuning deeper layers...")
for layer in model.layers:
    if 'block5' in layer.name or 'block4' in layer.name:
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[checkpoint, reduce_lr, early, csv_logger]
)

# ========================
# Final evaluation
# ========================
print("Evaluating on test set...")
model.load_weights(os.path.join(MODEL_DIR, 'best_model.h5'))
results = model.evaluate(test_gen)
print('Test results:', results)

# Save final
model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
print("Training complete. Model saved at:", os.path.join(MODEL_DIR, 'final_model.h5'))
