import os

DATA_DIR = "data/full_dataset/color"

classes = os.listdir(DATA_DIR)
print("Number of classes:", len(classes))

total_images = sum(
    len(files) for _, _, files in os.walk(DATA_DIR)
)

print("Total images:", total_images)