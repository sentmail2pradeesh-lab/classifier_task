import cv2
import numpy as np
import rawpy
import os
import io

EXPOSURE_LABELS = [
    #"Extra_Dark",
    "Dark",
    "Medium",
    "Bright",
    #"Extra_Bright"
]

LABEL_COLORS = {
    #"Extra_Dark": "#1a1a2e",
    "Dark": "#4a4e69",
    "Medium": "#9a8c98",
    "Bright": "#f2e9e4",
    #"Extra_Bright": "#ffd60a",
}


def compute_brightness(image_bytes: bytes, filename: str) -> float:
    
    extension = "." + filename.rsplit(".", 1)[-1].lower()

    raw_formats = {
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".raf",
        ".dng"
    }

    # -------------------------
    # RAW Image
    # -------------------------
    if extension in raw_formats:

        with rawpy.imread(io.BytesIO(image_bytes)) as raw:

            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8
            )

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        return float(np.mean(gray))

    # -------------------------
    # JPG / PNG
    # -------------------------
    arr = np.frombuffer(image_bytes, dtype=np.uint8)

    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(np.mean(gray))


def assign_label(index: int, total: int) -> str:
    if total == 1:
        return "Medium"

    if total == 2:
        return EXPOSURE_LABELS[index]   # Dark, Bright

    label_idx = min(2, (index * 3) // total)

    return EXPOSURE_LABELS[label_idx]


def classify_images(images: list[dict]) -> list[dict]:
    sorted_images = sorted(images, key=lambda x: x["brightness"])
    total = len(sorted_images)
    results = []

    for index, item in enumerate(sorted_images):
        label = assign_label(index, total)
        extension = "." + item["filename"].rsplit(".", 1)[-1].lower()

        raw_formats = {
            ".cr2",
            ".cr3",
            ".nef",
            ".arw",
            ".raf",
            ".dng"
        }

        if extension in raw_formats:
            extension = ".jpg"

        results.append(
        {
            **item,
            "label": label,
            "renamed_filename": f"{label}{extension}",
            "rank": index + 1,
        }
        )
    return results