import argparse
from collections import Counter, deque
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

try:
    from OneEuroFilter import OneEuroFilter
except ImportError:
    class OneEuroFilter:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, x, timestamp=None):
            return x

from sibi_gesture_utils import (
    HAND_FEATURE_SIZE,
    create_mediapipe_hands,
    landmarks_to_feature,
)

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferensi realtime kata SIBI dari video asli menggunakan MediaPipe + LSTM (TensorFlow/Keras)."
    )
    parser.add_argument(
        "--lstm-model-dir",
        default=str(PROJECT_DIR / "runs_sibi_lstm" / "exp1"),
        help="Folder tempat best.keras dan model_meta.json berada.",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Index kamera (misal: 0) atau path ke file video (misal: video.mp4).",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.60,
        help="Confidence minimum LSTM agar label ditampilkan.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Lebar target kamera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Tinggi target kamera.",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=10,
        help="Jumlah prediksi terakhir untuk smoothing label.",
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
        "--no-mirror",
        action="store_true",
        help="Matikan mirror horizontal preview.",
    )
    parser.add_argument(
        "--use-tflite",
        action="store_true",
        help="Gunakan model TFLite (best.tflite) alih-alih model Keras (best.keras).",
    )
    return parser.parse_args()


def load_model(model_dir: Path, use_tflite: bool):
    meta_path = model_dir / "model_meta.json"
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata model tidak ditemukan di {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    class_names = meta["class_names"]
    sequence_length = meta["sequence_length"]
    
    if use_tflite:
        model_path = model_dir / "best.tflite"
        if not model_path.exists():
            raise FileNotFoundError(f"Model TFLite tidak ditemukan di {model_path}")
        print("Memuat model TFLite...")
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter, class_names, sequence_length, True
    else:
        model_path = model_dir / "best.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model Keras tidak ditemukan di {model_path}")
        print("Memuat model Keras...")
        model = tf.keras.models.load_model(model_path)
        return model, class_names, sequence_length, False


def majority_vote(labels: deque[str]) -> str:
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


def count_detected_hands(feature: np.ndarray) -> int:
    left_detected = int(np.any(feature[:HAND_FEATURE_SIZE]))
    right_detected = int(np.any(feature[HAND_FEATURE_SIZE:]))
    return left_detected + right_detected


def process_sentence_buffer(words: list[str]) -> list[str]:
    if not words:
        return []
        
    # 1. Gabungkan huruf tunggal (spelling) menjadi satu kata
    merged_words = []
    current_spelling = []
    
    for word in words:
        # Asumsikan huruf tunggal alphabet A-Z
        if len(word) == 1 and word.isalpha():
            current_spelling.append(word.upper())
        else:
            if current_spelling:
                merged_words.append("".join(current_spelling))
                current_spelling = []
            merged_words.append(word)
            
    if current_spelling:
        merged_words.append("".join(current_spelling))
        
    # 2. Aturan penggabungan imbuhan berdasarkan request
    # Gunakan label persis seperti yang dikeluarkan model (misal: "KERAS", "AKHIRAN_NYA")
    rules = [
        (["AWALAN_ME", "CINTA", "AKHIRAN_I"], ["MENCINTAI"]),
        (["AWALAN_TER", "KENANG"], ["TERKENANG"]),
        (["AWALAN_DI", "SINI"], ["DISINI"]),
        (["AWALAN_ME", "DUKUNG"], ["MENDUKUNG"]),
        (["AWALAN_ME", "TOLONG"], ["MENOLONG"]),
        (["AWALAN_ME", "RAWAT"], ["MERAWAT"]),
        (["SEMUA", "AKHIRAN_NYA"], ["SEMUANYA"]),
        (["KERAS", "AKHIRAN_NYA"], ["KERASNYA"])
    ]
    
    def replace_sublist(lst, sublist, replacement):
        n = len(sublist)
        i = 0
        while i <= len(lst) - n:
            if lst[i:i+n] == sublist:
                lst[i:i+n] = replacement
                i += len(replacement)
            else:
                i += 1
        return lst

    for sublist, replacement in rules:
        merged_words = replace_sublist(merged_words, sublist, replacement)
        
    return merged_words



def main() -> None:
    args = parse_args()

    model_dir = Path(args.lstm_model_dir)
    lstm_model, class_names, sequence_length, is_tflite = load_model(model_dir, args.use_tflite)

    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    if isinstance(camera_source, int):
        cap = cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_source)
    else:
        cap = cv2.VideoCapture(camera_source)
        
    if not cap.isOpened():
        raise RuntimeError(f"Sumber video/kamera '{args.camera}' tidak bisa dibuka.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Buffers and State
    feature_buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
    prediction_buffer: deque[str] = deque(maxlen=max(1, args.smoothing))
    
    # Sentence Translation State
    raw_sentence: list[str] = []
    last_added_word: str = ""
    idle_frames: int = 0
    IDLE_RESET_WORD_FRAMES: int = 30  # Roughly 1 second of no hands to allow repeating the same word
    IDLE_CLEAR_SENTENCE_FRAMES: int = 90  # Roughly 3 seconds of no hands to clear the entire sentence

    window_name = "SIBI Recognizer (TensorFlow + OneEuroFilter)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)
    print("Webcam aktif. Tekan 'q' untuk keluar.")

    filter_freq = 30.0
    euro_filter = OneEuroFilter(
        freq=filter_freq, 
        mincutoff=1.0, 
        beta=0.001, 
        dcutoff=1.0
    )
    last_frame_time = time.time()

    with create_mediapipe_hands(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        max_num_hands=2,
        model_complexity=1,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_time = time.time()
            dt = current_time - last_frame_time
            last_frame_time = current_time

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmark_results = hands.process(rgb_frame)
            h, w = rgb_frame.shape[:2]
            raw_feature = landmarks_to_feature(landmark_results, image_width=w, image_height=h)
            
            try:
                feature = euro_filter(raw_feature, timestamp=current_time)
            except Exception:
                feature = raw_feature
                
            detected_hands = count_detected_hands(feature)
            feature_buffer.append(feature)

            predicted_label = ""
            predicted_confidence = 0.0

            # Idle Detection Logic
            if detected_hands == 0:
                idle_frames += 1
                if idle_frames == IDLE_RESET_WORD_FRAMES:
                    last_added_word = ""  # Allow user to sign the same word again
                    
                if idle_frames == IDLE_CLEAR_SENTENCE_FRAMES:
                    raw_sentence.clear()
                    prediction_buffer.clear()
            else:
                idle_frames = 0

            # Prediction Logic
            if len(feature_buffer) == sequence_length:
                input_tensor = np.stack(feature_buffer, axis=0)[np.newaxis, ...].astype(np.float32)
                
                # Dinamis: Cek apakah model membutuhkan Delta Features (252 neuron) atau model lama (126 neuron)
                if is_tflite:
                    expected_feature_size = lstm_model.get_input_details()[0]['shape'][-1]
                else:
                    expected_feature_size = lstm_model.input_shape[-1]
                    
                if expected_feature_size == 252 and input_tensor.shape[-1] == 126:
                    # Tambahkan Delta Features (Velocity)
                    deltas = np.zeros_like(input_tensor)
                    deltas[:, 1:, :] = input_tensor[:, 1:, :] - input_tensor[:, :-1, :]
                    input_tensor = np.concatenate([input_tensor, deltas], axis=-1)
                
                if is_tflite:
                    input_details = lstm_model.get_input_details()[0]
                    output_details = lstm_model.get_output_details()[0]
                    lstm_model.set_tensor(input_details['index'], input_tensor)
                    lstm_model.invoke()
                    probabilities = lstm_model.get_tensor(output_details['index'])[0]
                else:
                    probabilities = lstm_model.predict(input_tensor, verbose=0)[0]
                
                best_index = int(np.argmax(probabilities))
                predicted_confidence = float(probabilities[best_index])
                
                if predicted_confidence >= args.prediction_threshold:
                    predicted_label = class_names[best_index]
                    prediction_buffer.append(predicted_label)
                    
                    # Sentence Builder Logic: Hanya terima kalau buffer penuh dan label mendominasi
                    if len(prediction_buffer) == prediction_buffer.maxlen:
                        counts = Counter(prediction_buffer)
                        most_common_label, count = counts.most_common(1)[0]
                        
                        # Syarat: Label harus muncul minimal 60% dari buffer smoothing
                        required_count = int(0.6 * prediction_buffer.maxlen)
                        
                        if count >= required_count:
                            stable_label = most_common_label
                            if stable_label and stable_label != last_added_word:
                                raw_sentence.append(stable_label)
                                last_added_word = stable_label

                else:
                    prediction_buffer.clear()

            # --- Drawing UI ---
            
            # Top UI: Sentence Display
            display_sentence = process_sentence_buffer(raw_sentence)
            sentence_text = " ".join(display_sentence)
            if not sentence_text:
                sentence_text = "Menunggu isyarat..."
                
            cv2.putText(
                frame,
                sentence_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )
            
            # Bottom UI: Stats & Debug Info
            stats_y = frame.shape[0] - 20
            
            detecting_text = f"Deteksi Aktif: {predicted_label} (Conf: {predicted_confidence:.2f})"
            if idle_frames > 0:
                detecting_text = f"Diam (Idle: {idle_frames}/{IDLE_CLEAR_SENTENCE_FRAMES})"
                
            cv2.putText(
                frame,
                detecting_text,
                (20, stats_y - 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            cv2.putText(
                frame,
                f"Tangan: {detected_hands} | Buffer: {len(feature_buffer)}/{sequence_length}",
                (20, stats_y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            cv2.putText(
                frame,
                "Tekan 'q' keluar, 'c' hapus kalimat",
                (20, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                raw_sentence.clear()
                prediction_buffer.clear()
                last_added_word = ""
                idle_frames = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
