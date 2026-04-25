import os

def count_files_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

# Paths to the directories
train_normal = 'chest_xray/train/NORMAL'
train_pneumonia = 'chest_xray/train/PNEUMONIA'
test_normal = 'chest_xray/test/NORMAL'
test_pneumonia = 'chest_xray/test/PNEUMONIA'
val_normal = 'chest_xray/val/NORMAL'
val_pneumonia = 'chest_xray/val/PNEUMONIA'

# Count images
print("Training set:")
print(f"Normal: {count_files_in_directory(train_normal)}")
print(f"Pneumonia: {count_files_in_directory(train_pneumonia)}")
print(f"Total: {count_files_in_directory(train_normal) + count_files_in_directory(train_pneumonia)}")

print("\nTest set:")
print(f"Normal: {count_files_in_directory(test_normal)}")
print(f"Pneumonia: {count_files_in_directory(test_pneumonia)}")
print(f"Total: {count_files_in_directory(test_normal) + count_files_in_directory(test_pneumonia)}")

print("\nValidation set:")
print(f"Normal: {count_files_in_directory(val_normal)}")
print(f"Pneumonia: {count_files_in_directory(val_pneumonia)}")
print(f"Total: {count_files_in_directory(val_normal) + count_files_in_directory(val_pneumonia)}")
