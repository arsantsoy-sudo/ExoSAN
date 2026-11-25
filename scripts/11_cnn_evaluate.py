"""ExoSAN: оценка CNN и сохранение графиков. Запуск: python scripts/cnn_evaluate.py --mode fusion"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")  # Графики сохраняются в файлы, отдельное окно не открывается.
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

# Используем ТОТ ЖЕ загрузчик и те же классы, что во время обучения.
from cnn_model import ROOT, MODEL_DIR, REPORT_DIR, CLASSES, load_data, inputs_for

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["bls", "fusion"], default="fusion")
parser.add_argument("--split", choices=["val", "test"], default="val")
args = parser.parse_args()

# TEST используем только после выбора окончательной модели по validation!
rows, g, l, f, y = load_data(args.split)
if args.mode == "fusion":
    with np.load(MODEL_DIR / "tensorflow_cnn_fusion_scaler.npz") as s:
        f = np.clip((f - s["mean"]) / s["scale"], -5, 5).astype("float32")
model = keras.models.load_model(MODEL_DIR / f"tensorflow_cnn_{args.mode}.keras")
logits = model.predict(inputs_for(args.mode, g, l, f), batch_size=256, verbose=0)
prob = keras.activations.softmax(logits, axis=1).numpy()
pred = prob.argmax(axis=1)

report = classification_report(y, pred, labels=range(5), target_names=CLASSES,
                               output_dict=True, zero_division=0)
report["balanced_accuracy"] = balanced_accuracy_score(y, pred)
prefix = REPORT_DIR / f"tensorflow_cnn_{args.mode}_{args.split}"
REPORT_DIR.mkdir(exist_ok=True)
print(f"Balanced accuracy: {report['balanced_accuracy']:.4f}")
print(f"Macro-F1: {report['macro avg']['f1-score']:.4f}")

# Полные числовые результаты: удобны для анализа ошибок по KIC.
with open(str(prefix) + "_predictions.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["kepid", "true", "predicted", *["prob_" + c for c in CLASSES]])
    for r, t, p, probs in zip(rows, y, pred, prob):
        writer.writerow([r["kepid"], CLASSES[t], CLASSES[p], *probs.tolist()])
with open(str(prefix) + "_metrics.json", "w", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

# Матрица ошибок: строки = истинные классы, столбцы = предсказания.
cm = confusion_matrix(y, pred, labels=range(5))
with open(str(prefix) + "_confusion.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["true/predicted", *CLASSES])
    for name, values in zip(CLASSES, cm):
        writer.writerow([name, *values])
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="Blues")
fig.colorbar(im, ax=ax)
ax.set_xticks(range(5), CLASSES, rotation=30, ha="right")
ax.set_yticks(range(5), CLASSES)
ax.set_xlabel("Предсказанный класс")
ax.set_ylabel("Истинный класс")
ax.set_title(f"ExoSAN {args.mode}: confusion matrix ({args.split})")
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.tight_layout()
fig.savefig(str(prefix) + "_confusion.png", dpi=180)
plt.close(fig)

# Отдельный график качества распознавания каждого класса.
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(CLASSES, [report[c]["f1-score"] for c in CLASSES])
ax.set_ylim(0, 1)
ax.set_ylabel("F1")
ax.set_title(f"ExoSAN {args.mode}: F1 по классам ({args.split})")
ax.tick_params(axis="x", labelrotation=25)
fig.tight_layout()
fig.savefig(str(prefix) + "_f1.png", dpi=180)
plt.close(fig)

# Графики обучения строим по CSV, который создал cnn_model.py.
history_path = REPORT_DIR / f"tensorflow_cnn_{args.mode}_history.csv"
if history_path.exists():
    with history_path.open(newline="") as file:
        history = list(csv.DictReader(file))
    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(r["train_loss"]) for r in history], label="Train")
    axes[0].plot(epochs, [float(r["val_loss"]) for r in history], label="Validation")
    axes[0].set_title("Ошибка (loss)")
    axes[0].legend()
    axes[1].plot(epochs, [float(r["val_macro_f1"]) for r in history])
    axes[1].set_title("Validation Macro-F1")
    for ax in axes:
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(str(REPORT_DIR / f"tensorflow_cnn_{args.mode}_history.png"), dpi=180)
    plt.close(fig)

print(f"Все отчёты и графики сохранены в {REPORT_DIR}")
