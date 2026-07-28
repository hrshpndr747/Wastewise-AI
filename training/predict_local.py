from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

DEFAULT_THRESHOLD = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one waste image.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/best_model.keras"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("artifacts/labels.json"),
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Labels not found: {args.labels}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    with args.labels.open("r", encoding="utf-8") as file:
        labels: list[str] = json.load(file)

    model = tf.keras.models.load_model(args.model)
    height, width = model.input_shape[1:3]

    try:
        with Image.open(args.image) as image:
            image = image.convert("RGB").resize((width, height))
            array = np.asarray(image, dtype=np.float32)
    except UnidentifiedImageError as error:
        raise ValueError("The selected file is not a valid image.") from error

    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    ranked = np.argsort(probabilities)[::-1]
    best_index = int(ranked[0])
    confidence = float(probabilities[best_index])

    result = {
        "status": "success" if confidence >= args.threshold else "uncertain",
        "predicted_class": labels[best_index]
        if confidence >= args.threshold
        else None,
        "confidence": round(confidence, 4),
        "top_predictions": [
            {
                "class_name": labels[int(index)],
                "confidence": round(float(probabilities[int(index)]), 4),
            }
            for index in ranked[:3]
        ],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
