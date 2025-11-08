# model.py
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

def build_vgg16_model(input_shape=(224,224,3), n_classes=5, dropout=0.5, train_from_block=None):
    """
    Build VGG16-based classifier for DR detection.
    train_from_block: if None, only top layers are trained first.
    """

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(dropout)(x)
    predictions = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Fine-tuning option
    if train_from_block:
        set_trainable = False
        for layer in base_model.layers:
            if train_from_block in layer.name:
                set_trainable = True
            if set_trainable:
                layer.trainable = True

    return model
