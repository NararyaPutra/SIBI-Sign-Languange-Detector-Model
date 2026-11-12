import re
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_DIR / ".vendor"
VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
HAND_LANDMARK_COUNT = 21
HAND_FEATURE_SIZE = HAND_LANDMARK_COUNT * 3
SEQUENCE_FEATURE_SIZE = HAND_FEATURE_SIZE * 2
HAND_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_LANDMARKER_TASK_PATH = PROJECT_DIR / "hand_landmarker.task"


def normalize_sign_name(file_stem: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", file_stem.strip()).strip("_").upper()
    if not normalized:
        raise ValueError(f"Nama file '{file_stem}' tidak menghasilkan nama kelas yang valid.")
    return normalized


def parse_sign_classes(class_file: Path) -> list[str]:
    class_names: list[str] = []
    for raw_line in class_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        _, class_name = line.split(maxsplit=1)
        class_names.append(class_name.strip().upper())
    return class_names


def collect_video_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"File video tidak didukung: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Path input tidak ditemukan: {input_path}")

    return sorted(
        file_path
        for file_path in input_path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    )


def landmarks_to_feature(results, image_width: float = 1.0, image_height: float = 1.0) -> np.ndarray:
    feature = np.zeros((2, HAND_LANDMARK_COUNT, 3), dtype=np.float32)

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return feature.reshape(-1)

    # 1. Kumpulkan koordinat mentah terlebih dahulu tanpa dinormalisasi per tangan
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        hand_label = handedness.classification[0].label.lower()
        slot = 0 if hand_label == "left" else 1
        landmark_points = getattr(hand_landmarks, "landmark", hand_landmarks)

        # Ubah persentase menjadi piksel absolut (asli) agar bentuk tidak gepeng 
        # jika Aspect Ratio tidak 1:1. Nilai Z biasanya proporsional terhadap lebar (width).
        coords = np.array(
            [[landmark.x * image_width, landmark.y * image_height, landmark.z * image_width] for landmark in landmark_points],
            dtype=np.float32,
        )
        feature[slot] = coords

    # 2. Normalisasi global untuk mempertahankan jarak antara tangan kiri dan kanan
    active_mask = np.any(feature, axis=2)
    if not np.any(active_mask):
        return feature.reshape(-1)

    # Jadikan salah satu pergelangan (wrist) yang tersedia sebagai titik (0,0,0)
    if active_mask[0, 0]:
        origin = feature[0, 0].copy()
    else:
        origin = feature[1, 0].copy()

    # Geser semua titik aktif berdasarkan titik origin tersebut
    feature[active_mask] -= origin

    # Skalakan (scale) berdasarkan jarak rigid pada telapak tangan (Wrist ke Middle Finger MCP)
    palm_distances = []
    if active_mask[0, 0] and active_mask[0, 9]:
        palm_distances.append(np.linalg.norm(feature[0, 9, :2] - feature[0, 0, :2]))
    if active_mask[1, 0] and active_mask[1, 9]:
        palm_distances.append(np.linalg.norm(feature[1, 9, :2] - feature[1, 0, :2]))
        
    if palm_distances:
        scale = float(np.max(palm_distances))
    else:
        # Fallback jika landmark 0 atau 9 tidak terdeteksi aktif
        scale = float(np.max(np.linalg.norm(feature[active_mask][:, :2], axis=1)))

    if scale > 1e-6:
        feature[active_mask] /= scale

    return feature.reshape(-1)


@dataclass
class _SimpleLandmark:
    x: float
    y: float
    z: float


@dataclass
class _SimpleClassification:
    label: str


@dataclass
class _SimpleHandedness:
    classification: list[_SimpleClassification]


@dataclass
class _SimpleHandsResult:
    multi_hand_landmarks: list[list[_SimpleLandmark]]
    multi_handedness: list[_SimpleHandedness]


class _SolutionsHandsWrapper:
    def __init__(self, hands) -> None:
        self._hands = hands

    def __enter__(self):
        self._hands.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._hands.__exit__(exc_type, exc_val, exc_tb)

    def process(self, rgb_frame: np.ndarray):
        return self._hands.process(rgb_frame)


class _TasksHandsWrapper:
    def __init__(self, mp_module, landmarker) -> None:
        self._mp = mp_module
        self._landmarker = landmarker

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._landmarker.close()

    def process(self, rgb_frame: np.ndarray) -> _SimpleHandsResult:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(mp_image)

        multi_hand_landmarks: list[list[_SimpleLandmark]] = []
        multi_handedness: list[_SimpleHandedness] = []

        for index, hand_landmarks in enumerate(result.hand_landmarks):
            multi_hand_landmarks.append(
                [_SimpleLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks]
            )

            handedness_label = "Right"
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness_label = (
                    getattr(category, "category_name", None)
                    or getattr(category, "display_name", None)
                    or getattr(category, "label", None)
                    or "Right"
                )
            multi_handedness.append(_SimpleHandedness(classification=[_SimpleClassification(label=handedness_label)]))

        return _SimpleHandsResult(
            multi_hand_landmarks=multi_hand_landmarks,
            multi_handedness=multi_handedness,
        )


def ensure_hand_landmarker_task(model_asset_path: str | Path | None = None) -> Path:
    task_path = Path(model_asset_path) if model_asset_path else HAND_LANDMARKER_TASK_PATH
    if task_path.exists():
        return task_path

    task_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(HAND_LANDMARKER_TASK_URL, task_path)
    return task_path


@contextmanager
def _clean_import_environment(package_roots: tuple[str, ...] = ("matplotlib",)):
    original_sys_path = sys.path[:]
    removed_modules = {}

    blocked_paths = {
        str(PROJECT_DIR).lower(),
        str(VENDOR_DIR).lower(),
    }

    def is_blocked(path_entry: str) -> bool:
        try:
            normalized = str(Path(path_entry).resolve()).lower()
        except OSError:
            normalized = str(path_entry).lower()
        return normalized in blocked_paths or normalized.startswith(f"{str(VENDOR_DIR).lower()}\\")

    sys.path = [path_entry for path_entry in sys.path if not is_blocked(path_entry)]

    package_root_set = set(package_roots)
    for module_name in list(sys.modules):
        root_name = module_name.split(".", 1)[0]
        if root_name not in package_root_set:
            continue

        module = sys.modules[module_name]
        module_file = getattr(module, "__file__", "")
        if module_file and str(module_file).lower().startswith(str(VENDOR_DIR).lower()):
            removed_modules[module_name] = module
            del sys.modules[module_name]

    try:
        yield
    finally:
        sys.path = original_sys_path
        for module_name, module in removed_modules.items():
            if module_name not in sys.modules:
                sys.modules[module_name] = module


def _import_mediapipe_safely():
    with _clean_import_environment(("matplotlib",)):
        import mediapipe as mp
        return mp


def import_ultralytics_yolo():
    try:
        with _clean_import_environment(("ultralytics", "PIL", "matplotlib", "mpl_toolkits")):
            from ultralytics import YOLO
            return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Gagal mengimpor `ultralytics`. Penyebab paling sering adalah konflik dependency dari folder `.vendor` "
            "atau package `ultralytics`/`pillow` belum terpasang benar di env aktif."
        ) from exc


def build_sign_lstm_model(
    input_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    num_classes: int,
    dropout: float = 0.2,
    batch_size: int = None,
):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
    from tensorflow.keras.regularizers import l2

    model = Sequential()
    if batch_size is not None:
        model.add(Input(batch_shape=(batch_size, sequence_length, input_size)))
    else:
        model.add(Input(shape=(sequence_length, input_size)))
    
    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False
        model.add(Bidirectional(LSTM(hidden_size, return_sequences=return_seq, kernel_regularizer=l2(0.001))))
        if dropout > 0:
            model.add(Dropout(dropout))
            
    # Tambahan extra Dense layer dengan Batch Normalization
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
        
    model.add(Dense(num_classes, activation='softmax'))
    return model


def create_mediapipe_hands(
    min_detection_confidence: float = 0.4,
    min_tracking_confidence: float = 0.4,
    max_num_hands: int = 2,
    model_complexity: int = 1,
    model_asset_path: str | Path | None = None,
):
    try:
        mp = _import_mediapipe_safely()
    except ImportError as exc:
        raise RuntimeError(
            "Gagal mengimpor `mediapipe`. Penyebab paling sering adalah konflik dependency dari folder `.vendor` "
            "atau package `mediapipe` belum terpasang benar di env aktif."
        ) from exc

    if hasattr(mp, "solutions"):
        return _SolutionsHandsWrapper(
            mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity,
            )
        )

    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as exc:
        version = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            "Versi mediapipe di env aktif tidak mendukung API `mp.solutions.hands` "
            "dan modul `mediapipe.tasks.python` juga tidak bisa dipakai. "
            f"Versi terdeteksi: {version}."
        ) from exc

    task_path = ensure_hand_landmarker_task(model_asset_path)
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    return _TasksHandsWrapper(mp, landmarker)
