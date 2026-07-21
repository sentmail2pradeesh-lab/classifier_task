import base64
import io
import os
import zipfile
import tempfile
import rawpy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from classifier import LABEL_COLORS, classify_images, compute_brightness

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

OUTPUT_BASE = str(Path.home() / "Downloads")

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".cr2",
    ".cr3",
    ".nef",
    ".arw",
    ".raf",
    ".dng"
}


def allowed_file(filename: str) -> bool:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


def create_output_folder() -> tuple[str, str]:
    folder_name = datetime.now().strftime("classified_%Y%m%d_%H%M%S")
    folder_path = os.path.join(OUTPUT_BASE, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_name, folder_path


def save_image_to_disk(image_bytes, filename, output_path):
    
    extension = "." + filename.rsplit(".", 1)[-1].lower()

    raw_formats = {
        ".cr2",
        ".cr3",
        ".nef",
        ".arw",
        ".raf",
        ".dng"
    }

    # RAW Images
    if extension in raw_formats:

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        try:
            with rawpy.imread(temp_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8
                )

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imwrite(output_path, bgr)

        finally:
            os.remove(temp_path)

    # JPG / PNG / TIFF
    else:

        arr = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Unable to decode image")

        root, ext = os.path.splitext(output_path)

        if ext.lower() in raw_formats:
            output_path = root + ".jpg"
        cv2.imwrite(output_path, image)    


def unique_filename(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    stem, ext = os.path.splitext(base_name)
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{ext}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        counter += 1


def resolve_output_folder(folder_name: str) -> str | None:
    folder_path = os.path.join(OUTPUT_BASE, folder_name)
    if not os.path.isdir(folder_path):
        return None
    if os.path.commonpath([OUTPUT_BASE, os.path.abspath(folder_path)]) != os.path.abspath(OUTPUT_BASE):
        return None
    return folder_path

def create_preview(image_bytes, filename):
    
    extension = "." + filename.rsplit(".", 1)[-1].lower()

    raw_formats = {
        ".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng"
    }

    if extension in raw_formats:

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp:
            temp.write(image_bytes)
            temp_path = temp.name

        try:
            with rawpy.imread(temp_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=8
                )
        finally:
            os.remove(temp_path)

        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    else:

        arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    _, buffer = cv2.imencode(".jpg", image)

    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html", labels=LABEL_COLORS)


@app.route("/api/classify", methods=["POST"])
def classify():
    files = request.files.getlist("images")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No images uploaded"}), 400

    images = []
    errors = []

    for file in files:
        if not file.filename or not allowed_file(file.filename):
            errors.append(f"Skipped invalid file: {file.filename or 'unknown'}")
            continue

        data = file.read()
        try:
            brightness = compute_brightness(
                data,
                file.filename
)           
            images.append(
                {
                    "filename": file.filename,
                    "brightness": brightness,
                    "data": data,
                }
            )
        except ValueError:
            errors.append(f"Could not read image: {file.filename}")

    if not images:
        return jsonify({"error": "No valid images found", "warnings": errors}), 400

    classified = classify_images(
        [{"filename": img["filename"], "brightness": img["brightness"]} for img in images]
    )

    folder_name, folder_path = create_output_folder()
    data_by_filename = {img["filename"]: img["data"] for img in images}
    used_names: set[str] = set()
    results = []

    for item in classified:
        raw = data_by_filename[item["filename"]]
        renamed = unique_filename(item["renamed_filename"], used_names)
        output_path = os.path.join(folder_path, renamed)
        save_image_to_disk(
            raw,
            item["filename"],
            output_path
)

        preview = create_preview(raw, item["filename"])
        results.append(
            {
                "filename": item["filename"],
                "renamed_filename": renamed,
                "label": item["label"],
                "brightness": round(item["brightness"], 2),
                "rank": item["rank"],
                "color": LABEL_COLORS[item["label"]],
                "preview": preview,
                "saved_path": output_path,
            }
        )

    return jsonify(
        {
            "results": results,
            "warnings": errors,
            "output_folder": folder_name,
            "output_path": folder_path,
        }
    )


@app.route("/api/download", methods=["POST"])
def download():
    payload = request.get_json(silent=True)
    if not payload or "output_folder" not in payload:
        return jsonify({"error": "No output folder specified"}), 400

    folder_path = resolve_output_folder(payload["output_folder"])
    if not folder_path:
        return jsonify({"error": "Output folder not found"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, name)
            if os.path.isfile(file_path):
                zf.write(file_path, arcname=name)

    buf.seek(0)
    zip_name = f"{payload['output_folder']}.zip"
    zip_path = os.path.join(OUTPUT_BASE, zip_name)
    with open(zip_path, "wb") as f:
        f.write(buf.getvalue())
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=zip_name,
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
