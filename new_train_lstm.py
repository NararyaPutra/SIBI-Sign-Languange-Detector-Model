import argparse
import json
from pathlib import Path
import numpy as np


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from sibi_gesture_utils import build_sign_lstm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training LSTM untuk klasifikasi kata SIBI dari sequence landmark MediaPipe (menggunakan TensorFlow/Keras)."
    )
    parser.add_argument(
        "--dataset",
        default="sibi_sequence_dataset.npz",
        help="Path dataset sequence hasil builder (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        default="runs_sibi_lstm/exp1",
        help="Folder output model LSTM.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Jumlah epoch training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate optimizer Adam.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden size LSTM.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Jumlah layer LSTM.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout sesudah LSTM.",
    )
    parser.add_argument(
        "--noise-factor",
        type=float,
        default=0.015,
        help="Skala random Gaussian noise untuk augmentasi sequence landmark saat training.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Nilai label smoothing untuk categorical crossentropy (default: 0.1).",
    )
    parser.add_argument(
        "--drop-frame-prob",
        type=float,
        default=0.4,
        help="Probabilitas penerapan augmentasi random drop frame (simulasi lag/hold video call).",
    )
    parser.add_argument(
        "--delta-clip",
        type=float,
        default=0.05,
        help="Batas maksimum nilai delta koordinat (Delta Clamping) untuk mencegah velocity spike.",
    )
    return parser.parse_args()


class LandmarkDataGenerator(Sequence):
    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        num_classes=None,
        noise_factor=0.0,
        drop_frame_prob=0.0,
        delta_clip=0.05,
        shuffle=True,
    ):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.noise_factor = noise_factor
        self.drop_frame_prob = drop_frame_prob
        self.delta_clip = delta_clip
        self.shuffle = shuffle
        self.indices = np.arange(self.x.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def apply_temporal_stretch(self, sequence):
        if np.random.rand() > 0.5:
            return sequence
            
        seq_len = sequence.shape[0]
        factor = np.random.uniform(0.8, 1.2)
        new_len = int(seq_len * factor)
        new_len = max(5, min(new_len, 40))
        
        orig_steps = np.linspace(0, 1, seq_len)
        new_steps = np.linspace(0, 1, new_len)
        f = interp1d(orig_steps, sequence, axis=0, kind='linear', fill_value="extrapolate")
        stretched = f(new_steps)
        
        resampled_steps = np.linspace(0, 1, seq_len)
        f2 = interp1d(np.linspace(0, 1, new_len), stretched, axis=0, kind='linear', fill_value="extrapolate")
        return f2(resampled_steps)

    def apply_random_rotation(self, sequence):
        if np.random.rand() > 0.5:
            return sequence
            
        angle_x = np.random.uniform(-15, 15) * np.pi / 180.0
        angle_y = np.random.uniform(-15, 15) * np.pi / 180.0
        angle_z = np.random.uniform(-15, 15) * np.pi / 180.0
        
        cx, sx = np.cos(angle_x), np.sin(angle_x)
        cy, sy = np.cos(angle_y), np.sin(angle_y)
        cz, sz = np.cos(angle_z), np.sin(angle_z)
        
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        
        seq_3d = sequence.reshape(sequence.shape[0], -1, 3)
        rotated = np.dot(seq_3d, R.T)
        return rotated.reshape(sequence.shape[0], -1)

    def apply_spatial_dropout(self, sequence):
        if np.random.rand() > 0.1:
            return sequence
            
        seq_3d = sequence.reshape(sequence.shape[0], -1, 3).copy()
        
        num_points = seq_3d.shape[1]
        num_drop = np.random.randint(1, 4)
        
        drop_indices = np.random.choice(num_points, num_drop, replace=False)
        seq_3d[:, drop_indices, :] = 0.0
        
        return seq_3d.reshape(sequence.shape[0], -1)

    def apply_random_translation(self, sequence):
        if np.random.rand() > 0.5:
            return sequence
            
        seq_3d = sequence.reshape(sequence.shape[0], -1, 3).copy()
        
        shift_x = np.random.uniform(-0.02, 0.02)
        shift_y = np.random.uniform(-0.02, 0.02)
        shift_z = np.random.uniform(-0.02, 0.02)
        
        active_mask = np.any(seq_3d, axis=-1)
        seq_3d[active_mask, 0] += shift_x
        seq_3d[active_mask, 1] += shift_y
        seq_3d[active_mask, 2] += shift_z
        
        return seq_3d.reshape(sequence.shape[0], -1)

    def apply_random_drop_frame(self, sequence):
        if self.drop_frame_prob <= 0 or np.random.rand() > self.drop_frame_prob:
            return sequence
            
        seq = sequence.copy()
        seq_len = seq.shape[0]
        # Simulasikan 1 sampai 3 frame drop / hold untuk melatih ketahanan LSTM terhadap lag
        num_drops = np.random.randint(1, min(4, seq_len))
        drop_indices = np.random.choice(range(1, seq_len), size=num_drops, replace=False)
        
        for idx in sorted(drop_indices):
            seq[idx] = seq[idx - 1]  # Teknik Frame Hold
        return seq

    def compute_delta(self, batch_x):
        deltas = np.zeros_like(batch_x)
        deltas[:, 1:, :] = batch_x[:, 1:, :] - batch_x[:, :-1, :]
        if self.delta_clip > 0:
            # Kliping Nilai Delta (Delta Clamping) agar lonjakan kecepatan akibat VFR tidak merusak LSTM
            deltas = np.clip(deltas, -self.delta_clip, self.delta_clip)
        return np.concatenate([batch_x, deltas], axis=-1)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[batch_indices].copy()
        batch_y = self.y[batch_indices]
        
        if self.noise_factor > 0:
            for i in range(len(batch_x)):
                batch_x[i] = self.apply_temporal_stretch(batch_x[i])
                batch_x[i] = self.apply_random_rotation(batch_x[i])
                batch_x[i] = self.apply_random_drop_frame(batch_x[i])
                batch_x[i] = self.apply_spatial_dropout(batch_x[i])
                batch_x[i] = self.apply_random_translation(batch_x[i])
                
            noise = np.random.normal(0, self.noise_factor, batch_x.shape).astype(np.float32)
            active_mask = (batch_x != 0.0)
            batch_x[active_mask] += noise[active_mask]
            
        batch_x = self.compute_delta(batch_x)
        if self.num_classes is not None:
            batch_y = tf.keras.utils.to_categorical(batch_y, num_classes=self.num_classes)
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset sequence tidak ditemukan: {dataset_path}")

    dataset_json_path = dataset_path.with_suffix('.json')
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"Metadata JSON tidak ditemukan: {dataset_json_path}")
        
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
        
    class_names = meta_data["class_names"]
    sequence_length = int(meta_data["sequence_length"])
    feature_size = int(meta_data["feature_size"])

    data = np.load(dataset_path, allow_pickle=False)
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    X_valid = data["X_valid"].astype(np.float32)
    y_valid = data["y_valid"].astype(np.int64)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    if len(X_train) == 0:
        raise RuntimeError("Dataset train kosong. Bangun sequence dataset dulu sebelum training.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_feature_size = feature_size * 2 # Karena kita menambahkan Delta Features
    print(f"Membangun model LSTM dengan input_size={actual_feature_size}, sequence_length={sequence_length}, classes={len(class_names)}")
    model = build_sign_lstm_model(
        input_size=actual_feature_size,
        sequence_length=sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(class_names),
        dropout=args.dropout,
    )

    print(f"Menggunakan CategoricalCrossentropy (Label Smoothing={args.label_smoothing})...")
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    optimizer = Adam(learning_rate=args.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    best_model_path = output_dir / "best.keras"
    checkpoint_callback = ModelCheckpoint(
        filepath=str(best_model_path),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    print("Menghitung bobot kelas (Class Weights)...")
    class_weights_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights_array))

    print("Menyiapkan Data Generator dengan augmentasi...")
    print("Menyiapkan Data Generator dengan augmentasi (Random Drop Frame & Delta Clamping)...")
    num_classes = len(class_names)
    train_gen = LandmarkDataGenerator(
        X_train, y_train, 
        batch_size=args.batch_size, 
        num_classes=num_classes,
        noise_factor=args.noise_factor,
        drop_frame_prob=args.drop_frame_prob,
        delta_clip=args.delta_clip,
        shuffle=True
    )
    valid_gen = LandmarkDataGenerator(
        X_valid, y_valid,
        batch_size=args.batch_size,
        num_classes=num_classes,
        noise_factor=0.0,
        drop_frame_prob=0.0,
        delta_clip=args.delta_clip,
        shuffle=False
    )
    test_gen = LandmarkDataGenerator(
        X_test, y_test,
        batch_size=args.batch_size,
        num_classes=num_classes,
        noise_factor=0.0,
        drop_frame_prob=0.0,
        delta_clip=args.delta_clip,
        shuffle=False
    )

    print("Memulai proses training dengan Keras...")
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=args.epochs,
        class_weight=class_weights_dict,
        callbacks=[checkpoint_callback, early_stopping, reduce_lr]
    )

    print("\nMelakukan evaluasi pada set test...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    
    # sklearn metrics
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- Classification Report (Scikit-Learn) ---")
    labels = np.arange(len(class_names))
    report = classification_report(
        y_test, 
        y_pred, 
        labels=labels, 
        target_names=class_names,
        zero_division=0
    )
    print(report)
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(cm)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.jpg"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Save accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    acc_path = output_dir / "accuracy.jpg"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history.get('loss', []), label='Train Loss')
    plt.plot(history.history.get('val_loss', []), label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.tight_layout()
    loss_path = output_dir / "loss.jpg"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # Save training history and metadata
    history_path = output_dir / "history.json"
    
    # Clean numpy types for JSON serialization
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    
    history_path.write_text(
        json.dumps(
            {
                "history": history_dict,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    
    metadata_path = output_dir / "model_meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "input_size": feature_size,
                "sequence_length": sequence_length,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "drop_frame_prob": args.drop_frame_prob,
                "delta_clip": args.delta_clip,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nBest model   : {best_model_path.resolve()}")
    print(f"Model meta   : {metadata_path.resolve()}")
    print(f"History      : {history_path.resolve()}")
    print(f"Test loss    : {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\n--- Membangun model khusus TFLite (Batch Size = 1) ---")
    tflite_model = build_sign_lstm_model(
        input_size=actual_feature_size,
        sequence_length=sequence_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(class_names),
        dropout=args.dropout,
        batch_size=1
    )
    # Transfer bobot/weights dari model Keras yang sudah selesai di-training
    tflite_model.load_weights(best_model_path)
    
    print("Mengonversi ke TFLite Built-in Ops murni...")
    converter = tf.lite.TFLiteConverter.from_keras_model(tflite_model)
    
    # Memastikan tidak butuh "Select TF Ops" lagi, murni bawaan TFLite
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model_content = converter.convert()
    
    tflite_path = output_dir / "best.tflite"
    tflite_path.write_bytes(tflite_model_content)
    
    print(f"TFLite model  : {tflite_path.resolve()}")
    print("Export selesai!")


if __name__ == "__main__":
    main()
