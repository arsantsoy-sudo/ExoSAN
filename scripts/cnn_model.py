"""ExoSAN: обучение CNN-BLS или CNN-Fusion. Запуск: python scripts/cnn_model.py --mode fusion"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Папки проекта. Каталог действительно находится внутри data/catalogs.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CLASSES = ["planet", "false_positive", "eclipsing_binary", "stellar_variability", "quiet"]


def load_data(split):
    """Читаем список звёзд и уже рассчитанные фазовые кривые / Fourier-признаки."""
    with (ROOT / "data" / "catalogs" / "ml_split_v1.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    if len(rows) != {"train": 4200, "val": 900, "test": 900}[split]:
        raise ValueError(f"Неверное количество объектов в {split}: {len(rows)}")

    global_x, local_x, fft_x, labels = [], [], [], []
    for r in rows:
        kic = r["kepid"]
        with np.load(DATA / "views_v3" / f"{kic}.npz") as f:
            g = f["global_views"].astype("float32").copy()  # (3, 2, 1024)
            l = f["local_views"].astype("float32").copy()   # (3, 2, 256)
        with np.load(DATA / "fourier_v3" / f"{kic}.npz") as f:
            fft = f["feature_vector"].astype("float32")     # 48 признаков

        # Яркость переводим в ppt и преобразуем arcsinh; маска = 0 или 1.
        for view in (g, l):
            view[:, 0, :] = np.arcsinh(view[:, 0, :] * 1000)
            view[:, 1, :] = (view[:, 1, :] > 0).astype("float32")
        # TensorFlow ждёт (позиции, каналы), поэтому переставляем оси.
        global_x.append(g.reshape(6, 1024).T)
        local_x.append(l.reshape(6, 256).T)
        fft_x.append(fft)
        labels.append(CLASSES.index(r["class_label"]))

    return (rows, np.stack(global_x), np.stack(local_x),
            np.stack(fft_x), np.array(labels))


def branch(length, name):
    """Две свёртки: 16 и 32 фильтра; результат — 64 признака."""
    inp = keras.Input((length, 6))
    x = layers.Conv1D(16, 7, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    out = layers.Concatenate()([
        layers.GlobalAveragePooling1D()(x),
        layers.GlobalMaxPooling1D()(x),
    ])
    return keras.Model(inp, out, name=name)


def build_model(mode):
    """Та же архитектура, что в 09_full_cnn.py и 10_tensorflow_cnn.py."""
    g = keras.Input((1024, 6), name="global_input")
    l = keras.Input((256, 6), name="local_input")
    inputs = [g, l]
    features = [branch(1024, "global_branch")(g), branch(256, "local_branch")(l)]
    if mode == "fusion":
        f = keras.Input((48,), name="fourier_input")
        inputs.append(f)
        features.append(layers.Dense(16, activation="relu")(f))
    x = layers.Concatenate()(features)  # 128 признаков (BLS) или 144 (Fusion)
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dropout(0.20)(x)         # 20% признаков случайно отключаются при обучении
    out = layers.Dense(5)(x)            # Пять оценок классов (logits)
    model = keras.Model(inputs, out, name=f"ExoSAN_{mode}")
    model.compile(optimizer=keras.optimizers.AdamW(0.001, weight_decay=0.01),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    return model


def inputs_for(mode, g, l, f):
    return [g, l, f] if mode == "fusion" else [g, l]


class SaveBest(keras.callbacks.Callback):
    """По окончании каждой эпохи измеряем Macro-F1 и сохраняем лучшую модель."""
    def __init__(self, x_val, y_val, path):
        super().__init__()
        self.x_val, self.y_val, self.path = x_val, y_val, path
        self.best, self.rows = -1, []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = self.model.predict(self.x_val, batch_size=256, verbose=0).argmax(axis=1)
        f1 = f1_score(self.y_val, pred, average="macro", zero_division=0)
        self.rows.append([epoch + 1, logs["loss"], logs["val_loss"], f1])
        print(f"Эпоха {epoch + 1}: validation Macro-F1 = {f1:.4f}")
        if f1 > self.best:
            self.best = f1
            self.model.save(self.path)
            print("  Лучшая модель сохранена")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bls", "fusion"], default="fusion")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    keras.utils.set_random_seed(42)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    _, g, l, f, y = load_data("train")
    _, gv, lv, fv, yv = load_data("val")

    if args.mode == "fusion":
        # Статистики Fourier вычисляем ТОЛЬКО по train — без утечки из val/test.
        scaler = StandardScaler().fit(f)
        np.savez(MODEL_DIR / "tensorflow_cnn_fusion_scaler.npz",
                 mean=scaler.mean_, scale=scaler.scale_)
        f = np.clip(scaler.transform(f), -5, 5).astype("float32")
        fv = np.clip(scaler.transform(fv), -5, 5).astype("float32")

    model = build_model(args.mode)
    model.summary()
    x_val = inputs_for(args.mode, gv, lv, fv)
    cb = SaveBest(x_val, yv, MODEL_DIR / f"tensorflow_cnn_{args.mode}.keras")
    model.fit(inputs_for(args.mode, g, l, f), y,
              validation_data=(x_val, yv), epochs=args.epochs,
              batch_size=64, shuffle=True, callbacks=[cb], verbose=2)

    # История обучения нужна второму скрипту для графиков.
    with (REPORT_DIR / f"tensorflow_cnn_{args.mode}_history.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1"])
        writer.writerows(cb.rows)
    print("Готово. Теперь запусти cnn_evaluate.py --split val")
