import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from sibi_gesture_utils import VIDEO_EXTENSIONS, normalize_sign_name


SPLIT_NAMES = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pisahkan video SIBI ke train/valid/test sebelum generate frame."
    )
    parser.add_argument(
        "--input",
        default="Video",
        help="Folder sumber video mentah. Bisa berisi subfolder class.",
    )
    parser.add_argument(
        "--output-dir",
        default="video_split",
        help="Folder output split video.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Rasio video train.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Rasio video valid.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Rasio video test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed random agar split konsisten.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Hapus isi folder output sebelum menulis split baru.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Jumlah train-ratio, valid-ratio, dan test-ratio harus 1.0")


def collect_videos_by_class(input_dir: Path) -> dict[str, list[Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input tidak ditemukan: {input_dir}")

    class_to_videos: dict[str, list[Path]] = defaultdict(list)

    for video_path in sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS):
        relative_parent = video_path.parent.relative_to(input_dir)
        if relative_parent.parts:
            class_name = normalize_sign_name(relative_parent.parts[0])
        else:
            class_name = normalize_sign_name(video_path.stem)
        class_to_videos[class_name].append(video_path)

    if not class_to_videos:
        raise FileNotFoundError(f"Tidak ada video yang ditemukan di: {input_dir}")

    return dict(class_to_videos)


def calculate_split_counts(total_items: int, train_ratio: float, valid_ratio: float, test_ratio: float) -> dict[str, int]:
    if total_items <= 0:
        return {"train": 0, "valid": 0, "test": 0}

    counts = {
        "train": int(total_items * train_ratio),
        "valid": int(total_items * valid_ratio),
        "test": int(total_items * test_ratio),
    }

    remainder = total_items - sum(counts.values())
    order = sorted(
        [("train", train_ratio), ("valid", valid_ratio), ("test", test_ratio)],
        key=lambda item: item[1],
        reverse=True,
    )
    index = 0
    while remainder > 0:
        counts[order[index % len(order)][0]] += 1
        remainder -= 1
        index += 1

    if total_items >= 3:
        for split_name in SPLIT_NAMES:
            if counts[split_name] == 0:
                donor = max(SPLIT_NAMES, key=lambda name: counts[name])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[split_name] += 1

    if counts["train"] == 0:
        donor = "valid" if counts["valid"] >= counts["test"] else "test"
        if counts[donor] > 0:
            counts[donor] -= 1
            counts["train"] += 1

    return counts


def split_videos_by_class(
    class_to_videos: dict[str, list[Path]],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    split_map: dict[str, dict[str, list[Path]]] = {}

    for index, class_name in enumerate(sorted(class_to_videos)):
        shuffled = class_to_videos[class_name][:]
        random.Random(seed + index).shuffle(shuffled)
        counts = calculate_split_counts(len(shuffled), train_ratio, valid_ratio, test_ratio)

        train_end = counts["train"]
        valid_end = train_end + counts["valid"]
        split_map[class_name] = {
            "train": shuffled[:train_end],
            "valid": shuffled[train_end:valid_end],
            "test": shuffled[valid_end:],
        }

    return split_map


def write_sign_classes(output_dir: Path, class_names: list[str]) -> None:
    (output_dir / "sign_classes.txt").write_text(
        "\n".join(f"{index} {class_name}" for index, class_name in enumerate(class_names)) + "\n",
        encoding="utf-8",
    )


def copy_split_videos(output_dir: Path, split_map: dict[str, dict[str, list[Path]]]) -> None:
    for class_name, per_split in split_map.items():
        for split_name, video_paths in per_split.items():
            class_dir = output_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for video_path in video_paths:
                target_path = class_dir / video_path.name
                shutil.copy2(video_path, target_path)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.valid_ratio, args.test_ratio)

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    class_to_videos = collect_videos_by_class(input_dir)
    class_names = sorted(class_to_videos)
    split_map = split_videos_by_class(
        class_to_videos=class_to_videos,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_split_videos(output_dir, split_map)
    write_sign_classes(output_dir, class_names)

    for split_name in SPLIT_NAMES:
        total_videos = sum(len(split_map[class_name][split_name]) for class_name in class_names)
        print(f"{split_name}: videos={total_videos}")
        for class_name in class_names:
            count = len(split_map[class_name][split_name])
            if count:
                print(f"  - {class_name}: {count}")

    print(f"Output split   : {output_dir.resolve()}")
    print(f"Sign classes   : {(output_dir / 'sign_classes.txt').resolve()}")


if __name__ == "__main__":
    main()
