import os

image_dir = "dataset #6/Images" #Path to the file
label_dir = "dataset #6/labels" #Path to the file

# Obtains all of the images
images = [
    f for f in os.listdir(image_dir)
    if f.endswith(".jpg")
]

deleted = 0

# Deletes if there is not label that matches with the image
for img in images:

    base = os.path.splitext(img)[0]
    label_path = os.path.join(label_dir, base + ".txt")

    if not os.path.exists(label_path):

        os.remove(os.path.join(image_dir, img))
        print(f"Deleted image (no detection): {img}")
        deleted += 1

print(f"\nDone. Deleted {deleted} images.")