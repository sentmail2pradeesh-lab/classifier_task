import cv2
import os

from classifier import EXPOSURE_LABELS, classify_images, compute_brightness

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results = []

for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(INPUT_FOLDER, file)
        with open(path, "rb") as f:
            brightness = compute_brightness(f.read())
        results.append({"filename": file, "brightness": brightness, "path": path})

classified = classify_images(results)

print("\nExposure Classification\n")

for item in classified:
    extension = os.path.splitext(item["filename"])[1]
    new_name = item["label"] + extension
    output_path = os.path.join(OUTPUT_FOLDER, new_name)

    image = cv2.imread(item["path"])
    cv2.imwrite(output_path, image)

    print(f"{item['filename']}  -->  {new_name}  (brightness: {item['brightness']:.1f})")
