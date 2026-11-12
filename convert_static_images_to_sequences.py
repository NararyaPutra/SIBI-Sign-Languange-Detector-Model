import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert folder of static images into fake video sequences for LSTM dataset.")
    parser.add_argument("--input-dir", required=True, help="Folder containing raw images organized by class (e.g., dataset/A/, dataset/B/)")
    parser.add_argument("--output-dir", default="sequence_frames", help="Output directory to merge with existing sequence_frames")
    parser.add_argument("--sequence-length", type=int, default=16, help="Number of times to duplicate each image")
    parser.add_argument("--split-ratio", type=str, default="0.7,0.2,0.1", help="Train, Valid, Test split ratio")
    parser.add_argument("--classes", nargs="+", default=None, help="Daftar kelas tertentu yang ingin diproses (misal: --classes A S). Jika kosong, semua kelas diproses.")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(42)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found.")
        return
        
    ratios = [float(x) for x in args.split_ratio.split(",")]
    if len(ratios) != 3 or round(sum(ratios), 4) != 1.0:
        print("Error: Split ratio must contain 3 values (train,valid,test) summing to 1.0. Example: 0.7,0.2,0.1")
        return
        
    train_r, valid_r, test_r = ratios
    
    classes = [d for d in input_dir.iterdir() if d.is_dir()]
    if args.classes:
        target_classes = {c.upper() for c in args.classes}
        classes = [d for d in classes if d.name.upper() in target_classes]

    if not classes:
        print(f"Error: No matching class directories found in {input_dir}")
        return
        
    print(f"Found {len(classes)} classes. Processing...")
    
    total_images_processed = 0
    
    for class_dir in tqdm(classes, desc="Classes"):
        class_name = class_dir.name.upper()
        
        # Get all images in this class
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not images:
            continue
            
        random.shuffle(images)
        
        # Calculate split indices
        num_images = len(images)
        train_end = int(num_images * train_r)
        valid_end = train_end + int(num_images * valid_r)
        
        splits = {
            "train": images[:train_end],
            "valid": images[train_end:valid_end],
            "test": images[valid_end:]
        }
        
        for split_name, split_images in splits.items():
            for img_path in split_images:
                # Create a "video" directory for this single image
                # Example: sequence_frames/train/A/img_001_static/
                video_name = f"{img_path.stem}_static"
                target_video_dir = output_dir / split_name / class_name / video_name
                target_video_dir.mkdir(parents=True, exist_ok=True)
                
                # Duplicate the image sequence_length times
                for i in range(1, args.sequence_length + 1):
                    target_frame_path = target_video_dir / f"frame_{i:04d}{img_path.suffix}"
                    shutil.copy2(img_path, target_frame_path)
                    
                total_images_processed += 1

    print(f"\nDone! Processed {total_images_processed} static images into fake sequences.")
    print(f"You can now run: python build_mediapipe_lstm_dataset.py --input-dir {output_dir}")

if __name__ == "__main__":
    main()
