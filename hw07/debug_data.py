import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading data files...")
train_data = np.load(os.path.join(script_dir, 'train_data_3class.npy'), allow_pickle=True).item()
val_data = np.load(os.path.join(script_dir, 'val_data_3class.npy'), allow_pickle=True).item()
test_data = np.load(os.path.join(script_dir, 'test_data_3class.npy'), allow_pickle=True).item()

print(f"Train: {len(train_data['paths'])} samples")
print(f"Val: {len(val_data['paths'])} samples")
print(f"Test: {len(test_data['paths'])} samples")

train_labels = train_data['labels']
print(f"\nTraining class distribution:")
print(f"  Viral (0): {train_labels.count(0)}")
print(f"  Bacterial (1): {train_labels.count(1)}")
print(f"  Normal (2): {train_labels.count(2)}")

print("\nDone!")