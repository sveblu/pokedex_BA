import os
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import pathlib

# === Paths ===
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]  # points to ml-training/
DATA_DIR = ROOT_DIR / "data" / "tiny-imagenet-200" / "train"

# === Hyperparameters ===
IMG_SIZE = (64, 64)   # Tiny ImageNet uses 64x64 images
BATCH_SIZE = 64
EPOCHS = 1

# === Data loading ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalize to [0,1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))

# === Simple CNN model ===
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(200, activation="softmax")  # Tiny ImageNet has 200 classes
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# === Training ===
history = model.fit(train_ds, epochs=EPOCHS)

# === Save model ===
CKPT_DIR = ROOT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
model.save(CKPT_DIR / "tiny_imagenet_cnn.keras")
print("✅ Model saved to checkpoints/")
