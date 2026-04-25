import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime

log_file = open('train_3class_log.txt', 'w')

def log(msg):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    log(f"GPU configured: {len(gpus)} GPU(s) available")
else:
    log("No GPU available, using CPU")

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=NUM_CLASSES)

def create_dataset(image_paths, labels, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

log("Loading data...")
train_data = np.load(os.path.join(script_dir, 'train_data_3class.npy'), allow_pickle=True).item()
val_data = np.load(os.path.join(script_dir, 'val_data_3class.npy'), allow_pickle=True).item()
test_data = np.load(os.path.join(script_dir, 'test_data_3class.npy'), allow_pickle=True).item()

train_paths = train_data['paths']
train_labels = train_data['labels']
val_paths = val_data['paths']
val_labels = val_data['labels']
test_paths = test_data['paths']
test_labels = test_data['labels']

log(f"Training samples: {len(train_paths)}")
log(f"  Viral: {train_labels.count(0)}, Bacterial: {train_labels.count(1)}, Normal: {train_labels.count(2)}")
log(f"Validation samples: {len(val_paths)}")
log(f"Test samples: {len(test_paths)}")

train_dataset = create_dataset(train_paths, train_labels, shuffle=True)
val_dataset = create_dataset(val_paths, val_labels, shuffle=False)
test_dataset = create_dataset(test_paths, test_labels, shuffle=False)

log("Creating model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

log("Model summary:")
string_list = []
model.summary(print_fn=lambda x: string_list.append(x))
model_summary = '\n'.join(string_list)
log(model_summary)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

log("\nStarting training...")
start_time = datetime.datetime.now()
log(f"Start time: {start_time}")

history = model.fit(
    train_dataset,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_dataset,
    validation_steps=max(1, len(val_paths) // BATCH_SIZE),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

end_time = datetime.datetime.now()
log(f"End time: {end_time}")
log(f"Training duration: {end_time - start_time}")

model.save(os.path.join(script_dir, 'trained_model_3class.keras'))
log("\nTrained model saved to trained_model_3class.keras")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'figures', 'training_curves_3class.png'))
plt.close()
log("Training curves saved to figures/training_curves_3class.png")

import pickle
with open(os.path.join(script_dir, 'training_history_3class.pkl'), 'wb') as f:
    pickle.dump(history.history, f)
log("Training history saved to training_history_3class.pkl")

log("\nTraining completed successfully!")
log_file.close()