import os

image_dir = "dataset/images" # Path to file
label_dir = "dataset/labels" # Path to file

images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

for idx, img_name in enumerate(images):
    base_name = f"frame_{idx:06d}"

    old_label = labels[idx]  # assumes same count/order
    old_label_path = os.path.join(label_dir, old_label)
    new_label_path = os.path.join(label_dir, base_name + ".txt")

    os.rename(old_label_path, new_label_path)

print("Renaming complete.")