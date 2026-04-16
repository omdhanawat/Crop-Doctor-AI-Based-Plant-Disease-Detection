import os
import shutil
import random
import json

random.seed(42)

SOURCE_DIR = "data/full_dataset/color"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

SPLIT_RATIO = (0.7, 0.15, 0.15)

def split_dataset():

    summary = {}

    for class_name in os.listdir(SOURCE_DIR):

        class_path = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * SPLIT_RATIO[0])
        val_end = train_end + int(total * SPLIT_RATIO[1])

        splits = {
            TRAIN_DIR: images[:train_end],
            VAL_DIR: images[train_end:val_end],
            TEST_DIR: images[val_end:]
        }

        summary[class_name] = {
            "total": total,
            "train": len(images[:train_end]),
            "val": len(images[train_end:val_end]),
            "test": len(images[val_end:])
        }

        for split_dir, split_images in splits.items():
            target_class_dir = os.path.join(split_dir, class_name)
            os.makedirs(target_class_dir, exist_ok=True)

            for img in split_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(target_class_dir, img)
                shutil.copy(src_path, dst_path)

    os.makedirs("results", exist_ok=True)

    with open("results/split_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Split completed. Summary saved to results/split_summary.json")


if __name__ == "__main__":
    split_dataset()