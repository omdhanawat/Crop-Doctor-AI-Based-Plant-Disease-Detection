import os

BASE_DIR = "data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

for split in ["train", "valid"]:
    split_path = os.path.join(BASE_DIR, split)
    classes = os.listdir(split_path)

    print(f"\n{split.upper()} SET")
    print("Number of classes:", len(classes))

    total_images = 0
    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        total_images += len(os.listdir(cls_path))

    print("Total images:", total_images)