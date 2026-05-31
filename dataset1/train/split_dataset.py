import os
import shutil
import random

source_dir = r"C:\Users\gokul\Downloads\archive\Fingerprint Dataset for Blood Group Classification"

train_dir = "dataset/train"
test_dir = "dataset/test"

classes = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

for cls in classes:
    src = os.path.join(source_dir, cls)

    train_cls = os.path.join(train_dir, cls)
    test_cls = os.path.join(test_dir, cls)

    os.makedirs(train_cls, exist_ok=True)
    os.makedirs(test_cls, exist_ok=True)

    images = [
    f for f in os.listdir(src)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
]
    random.shuffle(images)

    split = int(len(images) * 0.8)

    train_images = images[:split]
    test_images = images[split:]

    for img in train_images:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(train_cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(src, img),
            os.path.join(test_cls, img)
        )

print("Dataset split completed!")