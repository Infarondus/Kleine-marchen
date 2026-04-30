"""
Оценка всего датасета на сохранённых моделях
Загружает все 5 фолдов из папки models/, прогоняет каждое изображение
через ансамбль и показывает где модель ошибается.

Запуск: python evaluate.py
Результаты сохраняются в папку eval_results/
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import shutil

# CONFIG — должно совпадать с обучением
WITH_SREDNIK_DIR    = './selected'
WITHOUT_SREDNIK_DIR = './rejected'
MODELS_DIR          = './models'
OUTPUT_DIR          = './eval_results'

MODEL_NAME   = 'vit_small_patch14_dinov2.lvd142m'
IMAGE_SIZE   = 280
BATCH_SIZE   = 32
NUM_WORKERS  = 0
DROP_PATH_RATE = 0.1

DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

# Порог уверенности для сомнительных предсказаний
DOUBT_THRESHOLD = 0.70   # если макс. вероятность ниже — считаем сомнительным



EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


def get_val_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                      border_mode=0, value=[255, 255, 255]),
        A.Normalize(mean=DINO_MEAN, std=DINO_STD),
        ToTensorV2(),
    ])


def get_tta_transforms():
    base = [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                      border_mode=0, value=[255, 255, 255]),
    ]
    norm = [A.Normalize(mean=DINO_MEAN, std=DINO_STD), ToTensorV2()]
    return [
        A.Compose(base + norm),
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + norm),
        A.Compose(base + [A.VerticalFlip(p=1.0)] + norm),
        A.Compose([A.LongestMaxSize(max_size=int(IMAGE_SIZE * 1.05)),
                   A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                                 border_mode=0, value=[255, 255, 255]),
                   A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE)] + norm),
        A.Compose(base + [A.RandomBrightnessContrast(
                          brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + norm),
    ]


def build_model():
    backbone = timm.create_model(
        MODEL_NAME, pretrained=False, num_classes=0,
        drop_path_rate=DROP_PATH_RATE,
        img_size=IMAGE_SIZE, dynamic_img_size=True,
    )
    embed_dim = backbone.embed_dim
    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )

    class DinoClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            return self.head(self.backbone(x))

    return DinoClassifier(backbone, head)


def load_models(models_dir, device):
    """Загружает все сохранённые фолды."""
    model_files = sorted(Path(models_dir).glob('fold_*_dinov2.pth'))
    if not model_files:
        raise FileNotFoundError(f"Не найдено моделей в {models_dir}. "
                                f"Убедитесь что папка models/ находится рядом со скриптом.")

    models = []
    for path in model_files:
        checkpoint = torch.load(path, map_location=device)
        model = build_model().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        fold = checkpoint.get('fold', '?')
        acc  = checkpoint.get('tta_acc', 0)
        print(f"  Загружен fold {fold+1}: val TTA acc={acc:.2%}")

    return models


@torch.no_grad()
def predict_ensemble(models, image_path, tta_transforms, device):
    """
    Прогоняет одно изображение через все модели и все TTA.
    Возвращает усреднённые вероятности [p_no_srednik, p_srednik].
    """
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        print(f"  ⚠ Не удалось открыть {image_path}: {e}")
        return None

    all_probs = []
    for model in models:
        for tfm in tta_transforms:
            tensor = tfm(image=img)['image'].unsqueeze(0).to(device)
            logits = model(tensor).cpu()
            probs  = F.softmax(logits, dim=1).squeeze(0)
            all_probs.append(probs)

    return torch.stack(all_probs).mean(0)  # [2]


def collect_items(directory, label):
    if not os.path.exists(directory):
        return []
    return [
        (os.path.join(directory, f), label)
        for f in sorted(os.listdir(directory))
        if Path(f).suffix.lower() in EXTENSIONS
    ]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    print(f"Модель: {MODEL_NAME} | Размер: {IMAGE_SIZE}px\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Загружаем модели ──────────────────────────
    print("Загружаем модели...")
    models = load_models(MODELS_DIR, device)
    print(f"Загружено моделей: {len(models)}\n")

    tta_transforms = get_tta_transforms()

    # ── Собираем датасет ──────────────────────────
    items_pos = collect_items(WITH_SREDNIK_DIR,    label=1)
    items_neg = collect_items(WITHOUT_SREDNIK_DIR, label=0)
    all_items = items_neg + items_pos

    print(f"Датасет: с средником={len(items_pos)}, без средника={len(items_neg)}")
    print(f"Итого: {len(all_items)} изображений\n")

    # ── Предсказания ──────────────────────────────
    CLASS_NAMES = {0: 'без средника', 1: 'со средником'}

    results = []   # (path, true_label, pred_label, confidence, probs)
    errors  = []   # только ошибки
    doubts  = []   # сомнительные (уверенность < DOUBT_THRESHOLD)

    print("Прогоняем изображения через ансамбль...")
    for i, (path, true_label) in enumerate(all_items):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_items)}...")

        probs = predict_ensemble(models, path, tta_transforms, device)
        if probs is None:
            continue

        pred_label   = probs.argmax().item()
        confidence   = probs.max().item()
        is_correct   = (pred_label == true_label)
        is_doubtful  = (confidence < DOUBT_THRESHOLD)

        results.append({
            'path':       path,
            'filename':   os.path.basename(path),
            'true':       true_label,
            'pred':       pred_label,
            'true_name':  CLASS_NAMES[true_label],
            'pred_name':  CLASS_NAMES[pred_label],
            'conf':       confidence,
            'p0':         probs[0].item(),
            'p1':         probs[1].item(),
            'correct':    is_correct,
            'doubtful':   is_doubtful,
        })

        if not is_correct:
            errors.append(results[-1])
        if is_doubtful:
            doubts.append(results[-1])

    # ── Метрики ───────────────────────────────────
    true_labels = [r['true'] for r in results]
    pred_labels = [r['pred'] for r in results]

    acc  = accuracy_score(true_labels, pred_labels)
    f1   = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    prec = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    rec  = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    cm   = confusion_matrix(true_labels, pred_labels)

    print(f"\n{'='*60}")
    print(f"  ИТОГИ АНСАМБЛЯ ({len(models)} моделей × {len(tta_transforms)} TTA)")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.2%}  ({sum(r['correct'] for r in results)}/{len(results)} верно)")
    print(f"  F1-macro:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                      Предсказано")
    print(f"                   без средника  со средником")
    print(f"  Реально без ср.     {cm[0,0]:>4}          {cm[0,1]:>4}")
    print(f"  Реально со ср.      {cm[1,0]:>4}          {cm[1,1]:>4}")

    # Разбивка ошибок по типу
    false_pos = [r for r in errors if r['true'] == 0]  # без средника → предсказано со средником
    false_neg = [r for r in errors if r['true'] == 1]  # со средником → предсказано без средника

    print(f"\n  Ошибок всего: {len(errors)}")
    print(f"    Без средника приняты за «со средником»: {len(false_pos)}")
    print(f"    Со средником приняты за «без средника»: {len(false_neg)}")
    print(f"\n  Сомнительных (уверенность < {DOUBT_THRESHOLD:.0%}): {len(doubts)}")

    #Сохраняем результаты
    # 1. Полный отчёт CSV
    csv_path = os.path.join(OUTPUT_DIR, 'full_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('filename,true_label,predicted_label,correct,confidence,p_no_srednik,p_srednik,path\n')
        for r in sorted(results, key=lambda x: x['conf']):  # сортируем по уверенности
            f.write(f"{r['filename']},{r['true_name']},{r['pred_name']},"
                    f"{'да' if r['correct'] else 'НЕТ'},{r['conf']:.4f},"
                    f"{r['p0']:.4f},{r['p1']:.4f},{r['path']}\n")
    print(f"\n  Полный отчёт: {csv_path}")

    # 2. Копируем ошибочные изображения в папки для просмотра
    errors_dir = os.path.join(OUTPUT_DIR, 'errors')
    fp_dir = os.path.join(errors_dir, 'false_positive__без_средника_принят_за_со_средником')
    fn_dir = os.path.join(errors_dir, 'false_negative__со_средником_принят_за_без_средника')
    for d in [fp_dir, fn_dir]:
        os.makedirs(d, exist_ok=True)

    for r in false_pos:
        dst = os.path.join(fp_dir, f"conf{r['conf']:.2f}__{r['filename']}")
        try: shutil.copy2(r['path'], dst)
        except: pass

    for r in false_neg:
        dst = os.path.join(fn_dir, f"conf{r['conf']:.2f}__{r['filename']}")
        try: shutil.copy2(r['path'], dst)
        except: pass

    print(f"  Ошибочные картинки скопированы в: {errors_dir}")

    # 3. Сомнительные
    if doubts:
        doubts_dir = os.path.join(OUTPUT_DIR, 'doubtful')
        os.makedirs(doubts_dir, exist_ok=True)
        for r in doubts:
            label = 'ВЕРНО' if r['correct'] else 'ОШИБКА'
            dst = os.path.join(doubts_dir,
                               f"{label}__conf{r['conf']:.2f}__true_{r['true_name']}__{r['filename']}")
            try: shutil.copy2(r['path'], dst)
            except: pass
        print(f"  Сомнительные картинки: {doubts_dir}")

    # 4. Топ-20 самых неуверенных предсказаний
    print(f"\n  {'─'*58}")
    print(f"  ТОП-20 САМЫХ НЕУВЕРЕННЫХ ПРЕДСКАЗАНИЙ")
    print(f"  {'─'*58}")
    least_confident = sorted(results, key=lambda x: x['conf'])[:20]
    for r in least_confident:
        mark = '✓' if r['correct'] else '✗'
        print(f"  {mark} {r['filename'][:40]:<40} "
              f"conf={r['conf']:.2f}  "
              f"true={r['true_name']:<15} pred={r['pred_name']}")

    print(f"\n  Готово. Все результаты в папке: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
