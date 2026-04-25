import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASS_NAMES = ['Viral', 'Bacterial', 'Normal']

script_dir = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=NUM_CLASSES)

def create_dataset(image_paths, labels, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

print("Loading data...")
test_data = np.load(os.path.join(script_dir, 'test_data_3class.npy'), allow_pickle=True).item()
test_paths = test_data['paths']
test_labels = test_data['labels']

print(f"Test samples: {len(test_paths)}")
print(f"  Viral: {test_labels.count(0)}, Bacterial: {test_labels.count(1)}, Normal: {test_labels.count(2)}")

test_dataset = create_dataset(test_paths, test_labels)

print("Loading model...")
model = tf.keras.models.load_model(os.path.join(script_dir, 'trained_model_3class.keras'))

print("Evaluating model...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

print("\nGenerating predictions...")
predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.array(test_labels)

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(true_classes, predicted_classes)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('3-Class Confusion Matrix (Viral / Bacterial / Normal)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'figures', 'confusion_matrix_3class.png'), dpi=150)
plt.close()
print("\nConfusion matrix saved to figures/confusion_matrix_3class.png")

plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('3-Class Normalized Confusion Matrix (Viral / Bacterial / Normal)', fontsize=14)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'figures', 'confusion_matrix_3class_normalized.png'), dpi=150)
plt.close()
print("Normalized confusion matrix saved to figures/confusion_matrix_3class_normalized.png")

print("\nClass Distribution in Test Set:")
for i, class_name in enumerate(CLASS_NAMES):
    count = test_labels.count(i)
    print(f"{class_name}: {count} samples ({count/len(test_labels)*100:.1f}%)")