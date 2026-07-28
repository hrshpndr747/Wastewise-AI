from __future__ import annotations

import io
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(os.getenv("WASTEWISE_MODEL_PATH", "artifacts/best_model.keras"))
LABELS_PATH = Path(os.getenv("WASTEWISE_LABELS_PATH", "artifacts/labels.json"))
CONFIDENCE_THRESHOLD = float(os.getenv("WASTEWISE_CONFIDENCE_THRESHOLD", "0.80"))

RECOMMENDATIONS = {
    "cardboard": "Flatten it and keep it clean and dry before recycling.",
    "glass": "Rinse it and place it in the appropriate glass recycling stream.",
    "metal": "Rinse containers and recycle them with accepted metal items.",
    "paper": "Keep the paper clean and dry before recycling.",
    "plastic": "Check the local recycling symbol and rinse the container.",
}


class ModelService:
    def __init__(self) -> None:
        self._model: tf.keras.Model | None = None
        self._labels: list[str] | None = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._labels is not None

    def load(self) -> None:
        if self.loaded:
            return

        with self._lock:
            if self.loaded:
                return

            if not MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Trained model not found at {MODEL_PATH.resolve()}. "
                    "Run training/train.py first."
                )
            if not LABELS_PATH.is_file():
                raise FileNotFoundError(
                    f"Label file not found at {LABELS_PATH.resolve()}."
                )

            with LABELS_PATH.open("r", encoding="utf-8") as file:
                labels = json.load(file)

            if not isinstance(labels, list) or not labels:
                raise ValueError("labels.json must contain a non-empty list.")

            self._model = tf.keras.models.load_model(MODEL_PATH)
            self._labels = [str(label) for label in labels]

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        self.load()
        assert self._model is not None
        assert self._labels is not None

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                height, width = self._model.input_shape[1:3]
                image = image.resize((width, height))
                array = np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("The uploaded file is not a readable image.") from error

        probabilities = self._model.predict(
            np.expand_dims(array, axis=0),
            verbose=0,
        )[0]

        ranked = np.argsort(probabilities)[::-1]
        best_index = int(ranked[0])
        confidence = float(probabilities[best_index])
        best_label = self._labels[best_index]

        top_predictions = [
            {
                "class_name": self._labels[int(index)],
                "confidence": round(float(probabilities[int(index)]), 4),
            }
            for index in ranked[:3]
        ]

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status": "uncertain",
                "predicted_class": None,
                "confidence": round(confidence, 4),
                "recommendation": (
                    "Try another photo with one object, better lighting, "
                    "and a plain background."
                ),
                "top_predictions": top_predictions,
            }

        return {
            "status": "success",
            "predicted_class": best_label,
            "confidence": round(confidence, 4),
            "recommendation": RECOMMENDATIONS.get(
                best_label,
                "Follow your local waste-disposal guidelines.",
            ),
            "top_predictions": top_predictions,
        }


model_service = ModelService()
