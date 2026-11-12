import argparse
import csv
from pathlib import Path

import cv2

from sibi_gesture_utils import VIDEO_EXTENSIONS, normalize_sign_name


SPLIT_NAMES = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frame dari video split tanpa crop agar siap diproses MediaPipe."
    )
    parser.add_argument(
        "--input-dir",
        default="video_split",
        help="Folder video yang sudah dipisah ke train/valid/test.",
    )
    parser.add_argument(
        "--output-dir",
        default="sequence_frames",
        help="Folder output frame per split, class, dan video.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=2,
        help="Ambil 1 frame setiap N frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Batasi jumlah frame per video. Isi 0 untuk tanpa batas.",
    )
    return parser.parse_args()


def iter_split_videos(input_dir: Path) -> list[tuple[str, str, Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Folder input split tidak ditemukan: {input_dir}")

    items: list[tuple[str, str, Path]] = []
    for split_name in SPLIT_NAMES:
        split_dir = input_dir / split_name
        if not split_dir.exists():
            continue

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            class_name = normalize_sign_name(class_dir.name)
            for video_path in sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS):
                items.append((split_name, class_name, video_path))

    if not items:
        raise FileNotFoundError(
            "Tidak ada video split yang ditemukan. Pastikan output split berada pada train/valid/test."
        )

    return items


def process_video(
    split_name: str,
    class_name: str,
    video_path: Path,
    output_dir: Path,
    sample_every: int,
    max_frames: int,
) -> int:
    video_name = normalize_sign_name(video_path.stem)
    target_dir = output_dir / split_name / class_name / video_name
    target_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = target_dir / "metadata.csv"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    saved_frames = 0
    source_frame_id = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["split", "class_name", "video_name", "frame_name", "source_frame_id"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if source_frame_id % sample_every != 0:
                source_frame_id += 1
                continue

            saved_frames += 1
            frame_name = f"frame_{saved_frames:05d}.jpg"
            frame_path = target_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            writer.writerow([split_name, class_name, video_name, frame_name, source_frame_id])

            source_frame_id += 1
            if max_frames > 0 and saved_frames >= max_frames:
                break

    cap.release()
    return saved_frames


def main() -> None:
    args = parse_args()
    if args.sample_every <= 0:
        raise ValueError("--sample-every harus lebih besar dari 0.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    items = iter_split_videos(input_dir)

    total_frames = 0
    for split_name, class_name, video_path in items:
        saved_frames = process_video(
            split_name=split_name,
            class_name=class_name,
            video_path=video_path,
            output_dir=output_dir,
            sample_every=args.sample_every,
            max_frames=args.max_frames,
        )
        total_frames += saved_frames
        print(
            f"{split_name}/{class_name}/{video_path.name} -> "
            f"frames={saved_frames} output={(output_dir / split_name / class_name / normalize_sign_name(video_path.stem)).resolve()}"
        )

    print(f"Output frame   : {output_dir.resolve()}")
    print(f"Total frame    : {total_frames}")


if __name__ == "__main__":
    main()
