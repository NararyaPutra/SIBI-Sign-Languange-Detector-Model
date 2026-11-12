import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from sibi_gesture_utils import (
    SEQUENCE_FEATURE_SIZE,
    create_mediapipe_hands,
    landmarks_to_feature,
    parse_sign_classes,
)


SPLIT_NAMES = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bangun dataset sequence MediaPipe dari frame video yang sudah dipisah per split."
    )
    parser.add_argument(
        "--input-dir",
        default="sequence_frames",
        help="Folder frame hasil generate dengan struktur train/valid/test per class dan per video.",
    )
    parser.add_argument(
        "--output",
        default="sibi_sequence_dataset.npz",
        help="File output dataset sequence (.npz).",
    )
    parser.add_argument(
        "--class-file",
        default="",
        help="Opsional file sign_classes.txt untuk menjaga urutan kelas.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Jumlah frame per sequence.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Lompatan sliding window sequence.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.4,
        help="Confidence minimum deteksi MediaPipe Hands.",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.4,
        help="Confidence minimum tracking MediaPipe Hands.",
    )
    parser.add_argument(
        "--pad-short-videos",
        action="store_true",
        help="Pad video pendek agar tetap menghasilkan satu sequence.",
    )
    return parser.parse_args()


def list_split_video_dirs(input_dir: Path) -> dict[str, dict[str, list[Path]]]:
    split_map: dict[str, dict[str, list[Path]]] = {}

    for split_name in SPLIT_NAMES:
        split_dir = input_dir / split_name
        class_to_videos: dict[str, list[Path]] = defaultdict(list)

        if split_dir.exists():
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                video_dirs = sorted(path for path in class_dir.iterdir() if path.is_dir())
                if video_dirs:
                    class_to_videos[class_dir.name.upper()] = video_dirs

        split_map[split_name] = class_to_videos

    if not any(split_map[split_name] for split_name in SPLIT_NAMES):
        raise FileNotFoundError(
            "Folder input belum memiliki struktur split train/valid/test. "
            "Jalankan split video lalu generate frame lebih dulu."
        )

    return split_map


def resolve_class_names(split_map: dict[str, dict[str, list[Path]]], class_file: str) -> list[str]:
    available_classes = sorted(
        {
            class_name
            for split_classes in split_map.values()
            for class_name, video_dirs in split_classes.items()
            if video_dirs
        }
    )
    if not available_classes:
        raise RuntimeError("Tidak ada class yang memiliki video/frame untuk dibangun menjadi dataset.")

    if not class_file:
        return available_classes

    ordered = []
    for class_name in parse_sign_classes(Path(class_file)):
        upper_name = class_name.upper()
        if upper_name in available_classes:
            ordered.append(upper_name)

    for class_name in available_classes:
        if class_name not in ordered:
            ordered.append(class_name)
    return ordered


def load_frame_features(video_dir: Path, hands) -> list[np.ndarray]:
    image_paths = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    features: list[np.ndarray] = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        h, w = rgb_frame.shape[:2]
        features.append(landmarks_to_feature(results, image_width=w, image_height=h))

    return features


def build_sequences(
    frame_features: list[np.ndarray],
    sequence_length: int,
    stride: int,
    pad_short_videos: bool,
) -> list[np.ndarray]:
    if not frame_features:
        return []

    if len(frame_features) < sequence_length:
        if not pad_short_videos:
            return []
        padded = frame_features[:] + [np.zeros(SEQUENCE_FEATURE_SIZE, dtype=np.float32)] * (sequence_length - len(frame_features))
        return [np.stack(padded, axis=0).astype(np.float32)]

    sequences: list[np.ndarray] = []
    last_start = len(frame_features) - sequence_length
    for start_index in range(0, last_start + 1, stride):
        window = frame_features[start_index:start_index + sequence_length]
        sequences.append(np.stack(window, axis=0).astype(np.float32))

    if not sequences:
        sequences.append(np.stack(frame_features[:sequence_length], axis=0).astype(np.float32))
    return sequences


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input frame tidak ditemukan: {input_dir}")

    split_map = list_split_video_dirs(input_dir)
    class_names = resolve_class_names(split_map, args.class_file)
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    dataset = {split_name: {"X": [], "y": [], "video_ids": []} for split_name in SPLIT_NAMES}
    summary: dict[str, dict[str, int]] = {split_name: defaultdict(int) for split_name in SPLIT_NAMES}

    with create_mediapipe_hands(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        max_num_hands=2,
        model_complexity=1,
    ) as hands:
        for split_name in SPLIT_NAMES:
            for class_name in class_names:
                for video_dir in split_map[split_name].get(class_name, []):
                    frame_features = load_frame_features(video_dir, hands)
                    sequences = build_sequences(
                        frame_features=frame_features,
                        sequence_length=args.sequence_length,
                        stride=args.stride,
                        pad_short_videos=args.pad_short_videos,
                    )
                    if not sequences:
                        continue

                    video_id = f"{split_name}/{class_name}/{video_dir.name}"
                    class_index = class_to_index[class_name]
                    for sequence in sequences:
                        dataset[split_name]["X"].append(sequence)
                        dataset[split_name]["y"].append(class_index)
                        dataset[split_name]["video_ids"].append(video_id)
                    summary[split_name][class_name] += len(sequences)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_train=np.array(dataset["train"]["X"], dtype=np.float32),
        y_train=np.array(dataset["train"]["y"], dtype=np.int64),
        video_ids_train=np.array(dataset["train"]["video_ids"], dtype=object),
        X_valid=np.array(dataset["valid"]["X"], dtype=np.float32),
        y_valid=np.array(dataset["valid"]["y"], dtype=np.int64),
        video_ids_valid=np.array(dataset["valid"]["video_ids"], dtype=object),
        X_test=np.array(dataset["test"]["X"], dtype=np.float32),
        y_test=np.array(dataset["test"]["y"], dtype=np.int64),
        video_ids_test=np.array(dataset["test"]["video_ids"], dtype=object),
        class_names=np.array(class_names, dtype=object),
        sequence_length=np.array([args.sequence_length], dtype=np.int64),
        feature_size=np.array([SEQUENCE_FEATURE_SIZE], dtype=np.int64),
    )

    metadata = {
        "class_names": class_names,
        "class_to_index": class_to_index,
        "sequence_length": args.sequence_length,
        "feature_size": SEQUENCE_FEATURE_SIZE,
        "stride": args.stride,
        "splits": {
            split_name: {class_name: count for class_name, count in split_summary.items()}
            for split_name, split_summary in summary.items()
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: sequences={len(dataset[split_name]['X'])} "
            f"videos={len(set(dataset[split_name]['video_ids']))}"
        )
    print(f"Dataset sequence : {output_path.resolve()}")
    print(f"Metadata         : {metadata_path.resolve()}")
    print(f"Kelas aktif      : {', '.join(class_names)}")


if __name__ == "__main__":
    main()
