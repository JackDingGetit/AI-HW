import os
import numpy as np
from collections import Counter

data = np.load('train_data_3class.npy', allow_pickle=True).item()
print("Training data distribution:")
labels = data['labels']
print(f"Total: {len(labels)}")
print(f"Class 0 (Viral): {labels.count(0)}")
print(f"Class 1 (Bacterial): {labels.count(1)}")
print(f"Class 2 (Normal): {labels.count(2)}")

data = np.load('val_data_3class.npy', allow_pickle=True).item()
print("\nValidation data distribution:")
labels = data['labels']
print(f"Total: {len(labels)}")
print(f"Class 0 (Viral): {labels.count(0)}")
print(f"Class 1 (Bacterial): {labels.count(1)}")
print(f"Class 2 (Normal): {labels.count(2)}")

data = np.load('test_data_3class.npy', allow_pickle=True).item()
print("\nTest data distribution:")
labels = data['labels']
print(f"Total: {len(labels)}")
print(f"Class 0 (Viral): {labels.count(0)}")
print(f"Class 1 (Bacterial): {labels.count(1)}")
print(f"Class 2 (Normal): {labels.count(2)}")