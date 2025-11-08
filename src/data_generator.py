# data_generator.py

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_generators(train_df, valid_df, test_df,
                   train_dir, val_dir, test_dir,
                   batch_size=16, target_size=(224, 224)):
    """
    Creates train/validation/test generators from image folders + CSV labels.
    """

    # ===============================
    # Convert diagnosis to string (required for class_mode='categorical')
    # ===============================
    train_df['diagnosis'] = train_df['diagnosis'].astype(str)
    valid_df['diagnosis'] = valid_df['diagnosis'].astype(str)
    test_df['diagnosis']  = test_df['diagnosis'].astype(str)

    # ===============================
    # Append image file extension (e.g., .png)
    # ===============================
    train_df['id_code'] = train_df['id_code'].astype(str) + '.png'
    valid_df['id_code'] = valid_df['id_code'].astype(str) + '.png'
    test_df['id_code']  = test_df['id_code'].astype(str) + '.png'

    # ===============================
    # Data augmentation for training
    # ===============================
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Only rescale for validation and test
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # ===============================
    # Generators
    # ===============================
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="id_code",
        y_col="diagnosis",
        target_size=target_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=valid_df,
        directory=val_dir,
        x_col="id_code",
        y_col="diagnosis",
        target_size=target_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col="id_code",
        y_col="diagnosis",
        target_size=target_size,
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen
