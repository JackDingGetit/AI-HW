import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Define image size and batch size
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# Define paths
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(script_dir, 'chest_xray', 'test')

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important for evaluation
)

# Load the trained model
try:
    model = tf.keras.models.load_model(os.path.join(script_dir, 'trained_model.keras'))
    print("Model loaded successfully from trained_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Evaluate the model on test data
print("\nEvaluating model on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Get true labels
true_classes = test_generator.classes
class_indices = test_generator.class_indices
class_names = list(class_indices.keys())

# Calculate classification metrics
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Calculate confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(script_dir, 'confusion_matrix.png'))
print("\nConfusion matrix saved to confusion_matrix.png")
plt.show()

# Print class distribution
print("\nClass Distribution in Test Set:")
for class_name, class_idx in class_indices.items():
    count = np.sum(true_classes == class_idx)
    print(f"{class_name}: {count} samples")