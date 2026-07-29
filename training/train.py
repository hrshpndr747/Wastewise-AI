from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WasteWise CNN.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    seed: int,
):
    common = {
        "image_size": (image_size, image_size),
        "batch_size": batch_size,
        "label_mode": "int",
    }

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        shuffle=True,
        seed=seed,
        **common,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        shuffle=False,
        **common,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        shuffle=False,
        **common,
    )

    class_names = train_ds.class_names
    if class_names != val_ds.class_names or class_names != test_ds.class_names:
        raise ValueError("Class folders differ between train, val, and test sets.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def build_model(image_size: int, number_of_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="wastewise_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_curves(history: tf.keras.callbacks.History, output: Path) -> None:
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output.parent / "history.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history_frame["accuracy"], label="Training")
    axes[0].plot(history_frame["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history_frame["loss"], label="Training")
    axes[1].plot(history_frame["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_and_save(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    artifacts_dir: Path,
) -> dict[str, float]:
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        true_labels.extend(labels.numpy().tolist())
        predicted_labels.extend(predictions.tolist())

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        artifacts_dir / "classification_report.csv"
    )

    matrix = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    figure, axis = plt.subplots(figsize=(8, 8))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title("WasteWise Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(
        artifacts_dir / "confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    metrics = {
        "test_loss": round(float(test_loss), 6),
        "test_accuracy": round(float(test_accuracy), 6),
        "macro_precision": round(float(report["macro avg"]["precision"]), 6),
        "macro_recall": round(float(report["macro avg"]["recall"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
    }

    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    required_splits = [args.data_dir / name for name in ("train", "val", "test")]
    missing = [str(path) for path in required_splits if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Processed dataset is missing: "
            + ", ".join(missing)
            + ". Run training/prepare_dataset.py first."
        )

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_dir,
        args.image_size,
        args.batch_size,
        args.seed,
    )

    with (args.artifacts_dir / "labels.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    model = build_model(args.image_size, len(class_names))
    model.summary()

    checkpoint_path = args.artifacts_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # Reload the checkpoint selected by validation accuracy.
    model = tf.keras.models.load_model(checkpoint_path)

    save_training_curves(
        history,
        args.artifacts_dir / "training_curves.png",
    )
    metrics = evaluate_and_save(
        model,
        test_ds,
        class_names,
        args.artifacts_dir,
    )

    print("\nFinal test metrics")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"\nSaved artifacts to: {args.artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()
