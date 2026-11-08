"""
Честная out-of-fold (OOF) оценка DINOv2.

Каждое изображение оценивается ТОЛЬКО той моделью-фолдом, которая не видела
его при обучении (с теми же 5 TTA, что и при обучении). Разбиение на фолды
восстанавливается точно так же, как в deep_test_v4_dino.py:
  тот же порядок файлов (sorted(os.listdir), сначала selected, потом rejected)
  и StratifiedKFold(n_splits=5, shuffle=True, random_state=42).

Самопроверка: точность каждого фолда, пересчитанная здесь, должна почти
совпасть с tta_acc, сохранённым в чекпоинте при обучении. Если не совпадает,
значит, папки selected/rejected менялись после обучения и разбиение
восстановить нельзя (скрипт об этом предупредит).

Запуск: положить рядом с папками selected/, rejected/, models/ и выполнить
    python dino_oof.py
Результат: oof_results.csv + сводка в консоли (её и пришлите).
"""
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

# Должно совпадать с deep_test_v4_dino.py
WITH_SREDNIK_DIR    = './selected'
WITHOUT_SREDNIK_DIR = './rejected'
MODELS_DIR          = './models'
MODEL_GLOBS         = ['fold_*_km.pth', 'fold_*_dinov2.pth']
MODEL_NAME          = 'vit_small_patch14_dinov2.lvd142m'
IMAGE_SIZE          = 280
SEED                = 42
NUM_FOLDS           = 5
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def collect_items(directory, label):
    return [(os.path.join(directory, f), label)
            for f in sorted(os.listdir(directory))
            if os.path.splitext(f)[1].lower() in EXTENSIONS]


def tta_transforms():
    base = [A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
                          border_mode=0, value=[255, 255, 255])]
    norm = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
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
    backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0,
                                 drop_path_rate=0.1, img_size=IMAGE_SIZE,
                                 dynamic_img_size=True)
    d = backbone.embed_dim
    head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256, 2))

    class DinoClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            return self.head(self.backbone(x))

    return DinoClassifier()


def manuscript_key(path):
    """Нормализованный идентификатор рукописи: без conf-префикса, '_' вместо пробелов."""
    name = re.sub(r'^conf[\d.]+__', '', os.path.basename(path))
    return os.path.splitext(name.replace(' ', '_').lower())[0]


def fond_of(key):
    m = re.match(r'ор_ргб_(.+)_(\d+)$', key)
    return m.group(1) if m else '?'


def metrics(y, p):
    return dict(acc=accuracy_score(y, p),
                prec=precision_score(y, p, average='macro', zero_division=0),
                rec=recall_score(y, p, average='macro', zero_division=0),
                f1=f1_score(y, p, average='macro', zero_division=0),
                cm=confusion_matrix(y, p, labels=[0, 1]))


def show(title, m, n):
    cm = m['cm']
    print(f"\n  {title} (n={n})")
    print(f"    Accuracy {m['acc']:.4f} | Precision {m['prec']:.4f} | "
          f"Recall {m['rec']:.4f} | F1 {m['f1']:.4f}")
    print(f"    без средника: {cm[0,0]} верно / {cm[0,1]} ошибочно")
    print(f"    со средником: {cm[1,1]} верно / {cm[1,0]} ошибочно")
    print(f"    полнота по классам: без={cm[0,0]/cm[0].sum():.3f}, со={cm[1,1]/cm[1].sum():.3f}")


@torch.no_grad()
def predict(model, path, tfms, device):
    img = np.array(Image.open(path).convert('RGB'))
    probs = []
    for t in tfms:
        x = t(image=img)['image'].unsqueeze(0).to(device)
        probs.append(F.softmax(model(x).float().cpu(), dim=1).squeeze(0))
    return torch.stack(probs).mean(0)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"torch {torch.__version__} | CUDA в сборке: {torch.version.cuda} | устройство: "
          f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'ПРОЦЕССОР (CPU)'}")

    # Порядок как в обучении: items_pos + items_neg
    items = collect_items(WITH_SREDNIK_DIR, 1) + collect_items(WITHOUT_SREDNIK_DIR, 0)
    labels = [l for _, l in items]
    print(f"Изображений: {len(items)} | {Counter(labels)}")

    model_files = []
    for g in MODEL_GLOBS:
        model_files += sorted(Path(MODELS_DIR).glob(g))
    ckpts = {}
    for p in model_files:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        ckpts[ck['fold']] = (p.name, ck)
    print(f"Фолды в чекпоинтах: {sorted(ckpts)} -> {[ckpts[k][0] for k in sorted(ckpts)]}")
    assert sorted(ckpts) == list(range(NUM_FOLDS)), "Нужно ровно 5 чекпоинтов с fold=0..4"

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    tfms = tta_transforms()
    oof = {}
    fold_of = {}
    print("\nСамопроверка разбиения (пересчитано / сохранено при обучении):")
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        name, ck = ckpts[k]
        model = build_model().to(device)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        preds = []
        for n, i in enumerate(val_idx, 1):
            if n % 50 == 0:
                print(f"    fold {k+1}: {n}/{len(val_idx)}", flush=True)
            oof[i] = predict(model, items[i][0], tfms, device)
            fold_of[i] = k
            preds.append(int(oof[i].argmax()))
        acc = accuracy_score([labels[i] for i in val_idx], preds)
        saved = ck.get('tta_acc', float('nan'))
        flag = 'OK' if abs(acc - saved) < 0.015 else '!!! РАСХОЖДЕНИЕ'
        print(f"  fold {k+1} ({name}): {acc:.4f} / {saved:.4f}  {flag}")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    idx = sorted(oof)
    y = np.array([labels[i] for i in idx])
    p = np.array([int(oof[i].argmax()) for i in idx])
    conf = np.array([float(oof[i].max()) for i in idx])

    print("\n" + "=" * 64)
    print("  ЧЕСТНАЯ OOF-ОЦЕНКА DINOv2 (каждое изображение — моделью, не видевшей его)")
    print("=" * 64)
    show("Все изображения", metrics(y, p), len(y))

    # Дубли (одна рукопись дважды) и попадание копий в разные фолды
    keys = [manuscript_key(items[i][0]) for i in range(len(items))]
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    cross = [g for g in dup_groups if len({fold_of[i] for i in g}) > 1]
    print(f"\n  Уникальных рукописей: {len(groups)} из {len(items)}; "
          f"групп-дублей: {len(dup_groups)}, из них копии в РАЗНЫХ фолдах: {len(cross)}")
    first = sorted(g[0] for g in groups.values())
    show("Без дублей (по одной копии на рукопись)",
         metrics(np.array([labels[i] for i in first]),
                 np.array([int(oof[i].argmax()) for i in first])), len(first))

    # Разбивка по фондам
    print("\n  По фондам:")
    by_fond = defaultdict(list)
    for i in first:
        by_fond[fond_of(keys[i])].append(i)
    for f, ids in sorted(by_fond.items(), key=lambda kv: -len(kv[1])):
        yy = [labels[i] for i in ids]
        pp = [int(oof[i].argmax()) for i in ids]
        print(f"    фонд {f:>6}: n={len(ids):>4} | со/без = {sum(yy):>3}/{len(yy)-sum(yy):>3} | "
              f"accuracy {accuracy_score(yy, pp):.3f}")

    for thr in (0.55, 0.70):
        print(f"\n  Доля с уверенностью < {thr}: {(conf < thr).mean():.1%}")
    sel = np.array([float(oof[i][1]) for i in idx]) >= 0.55
    if sel.any():
        print(f"  Порог τ2=0.55 на p(со средником): отобрано {sel.sum()}, "
              f"precision отбора {(y[sel] == 1).mean():.4f}, "
              f"recall {(sel & (y == 1)).sum() / (y == 1).sum():.4f}")

    with open('oof_results.csv', 'w', encoding='utf-8') as f:
        f.write('path,manuscript,fond,fold,true,pred,p_srednik,confidence\n')
        for i in idx:
            f.write(f"{items[i][0]},{keys[i]},{fond_of(keys[i])},{fold_of[i]+1},"
                    f"{labels[i]},{int(oof[i].argmax())},{float(oof[i][1]):.4f},"
                    f"{float(oof[i].max()):.4f}\n")
    print("\n  Подробно по каждому изображению: oof_results.csv")


if __name__ == '__main__':
    main()
