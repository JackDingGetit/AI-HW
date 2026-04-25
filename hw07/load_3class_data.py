import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def get_three_class_label(filepath):
    filepath_lower = filepath.lower()
    if '_virus_' in filepath_lower:
        return 0
    elif '_bacteria_' in filepath_lower:
        return 1
    else:
        return 2

def load_data_from_dir(base_dir, img_size=(150, 150)):
    image_paths = []
    labels = []
    class_names_set = set()

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                filepath = os.path.join(root, filename)
                label = get_three_class_label(filepath)
                image_paths.append(filepath)
                labels.append(label)
                if label == 0:
                    class_names_set.add('Viral')
                elif label == 1:
                    class_names_set.add('Bacterial')
                else:
                    class_names_set.add('Normal')

    return image_paths, labels

def create_dataset(image_paths, labels, img_size=(150, 150), batch_size=32):
    def load_and_preprocess(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.one_hot(label, depth=3)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(script_dir, 'chest_xray', 'train')
test_dir = os.path.join(script_dir, 'chest_xray', 'test')
val_dir = os.path.join(script_dir, 'chest_xray', 'val')

print("Loading training data...")
train_paths, train_labels = load_data_from_dir(train_dir)
print(f"Training samples: {len(train_paths)}")
print(f"  Viral: {train_labels.count(0)}, Bacterial: {train_labels.count(1)}, Normal: {train_labels.count(2)}")

print("\nLoading validation data...")
val_paths, val_labels = load_data_from_dir(val_dir)
print(f"Validation samples: {len(val_paths)}")
print(f"  Viral: {val_labels.count(0)}, Bacterial: {val_labels.count(1)}, Normal: {val_labels.count(2)}")

print("\nLoading test data...")
test_paths, test_labels = load_data_from_dir(test_dir)
print(f"Test samples: {len(test_paths)}")
print(f"  Viral: {test_labels.count(0)}, Bacterial: {test_labels.count(1)}, Normal: {test_labels.count(2)}")

BATCH_SIZE = 32
IMG_SIZE = (150, 150)

print("\nCreating TensorFlow datasets...")
train_dataset = create_dataset(train_paths, train_labels, IMG_SIZE, BATCH_SIZE)
val_dataset = create_dataset(val_paths, val_labels, IMG_SIZE, BATCH_SIZE)
test_dataset = create_dataset(test_paths, test_labels, IMG_SIZE, BATCH_SIZE)

print("\nDatasets created successfully!")
print(f"Number of training batches: {len(train_dataset)}")
print(f"Number of validation batches: {len(val_dataset)}")
print(f"Number of test batches: {len(test_dataset)}")

np.save('train_data_3class.npy', {'paths': train_paths, 'labels': train_labels})
np.save('val_data_3class.npy', {'paths': val_paths, 'labels': val_labels})
np.save('test_data_3class.npy', {'paths': test_paths, 'labels': test_labels})
print("\nData saved to npy files for later use.")