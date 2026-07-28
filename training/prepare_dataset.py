from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

ALLOWED_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_files(files: Iterable[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files):
        target = destination / f"{index:05d}_{source.name}"
        shutil.copy2(source, target)


def split_class(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = files.copy()
    random.Random(seed).shuffle(shuffled)

    count = len(shuffled)
    train_end = int(count * train_ratio)
    val_end = train_end + int(count * val_ratio)

    # Keep every split non-empty for sufficiently large classes.
    if count >= 10:
        train_end = max(1, min(train_end, count - 2))
        val_end = max(train_end + 1, min(val_end, count - 1))

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and stratify WasteWise raw images."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate the processed dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("Training and validation ratios must be positive.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    if args.output_dir.exists():
        contains_files = any(path.is_file() for path in args.output_dir.rglob("*"))
        if contains_files and not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already contains files. "
                "Use --overwrite to rebuild it."
            )
        if args.overwrite:
            shutil.rmtree(args.output_dir)

    missing = [
        class_name
        for class_name in ALLOWED_CLASSES
        if not (args.raw_dir / class_name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing raw class folders: "
            + ", ".join(missing)
            + f". Expected them inside {args.raw_dir.resolve()}."
        )

    summary: dict[str, dict[str, int]] = {}

    for class_index, class_name in enumerate(ALLOWED_CLASSES):
        source_folder = args.raw_dir / class_name
        files = image_files(source_folder)

        if len(files) < 10:
            raise ValueError(
                f"Class '{class_name}' contains only {len(files)} valid images. "
                "Add at least 10 images; substantially more are recommended."
            )

        train_files, val_files, test_files = split_class(
            files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + class_index,
        )

        copy_files(train_files, args.output_dir / "train" / class_name)
        copy_files(val_files, args.output_dir / "val" / class_name)
        copy_files(test_files, args.output_dir / "test" / class_name)

        summary[class_name] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }

    print("\nDataset split complete\n")
    print(f"{'Class':<12}{'Total':>8}{'Train':>8}{'Val':>8}{'Test':>8}")
    print("-" * 44)
    for class_name, counts in summary.items():
        print(
            f"{class_name:<12}"
            f"{counts['total']:>8}"
            f"{counts['train']:>8}"
            f"{counts['val']:>8}"
            f"{counts['test']:>8}"
        )
    print(f"\nProcessed dataset: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
