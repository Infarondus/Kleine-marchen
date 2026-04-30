"""
Инференс: сортировка неразмеченных переплётов
Берёт папку с фотографиями и раскладывает по трём папкам:
  result/со_средником/     — уверенно со средником
  result/без_средника/     — уверенно без средника
  result/сомнительные/     — модель не уверена, нужен человек

Запуск:
  python inference.py
  python inference.py --input ./мои_фото --threshold 0.80
"""

import os
import argparse
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CONFIG
INPUT_DIR    = './training_option'        # папка с неразмеченными фото
OUTPUT_DIR   = './result'         # куда складывать результаты
MODELS_DIR   = './models'         # папка с обученными моделями

MODEL_NAME   = 'vit_small_patch14_dinov2.lvd142m'
IMAGE_SIZE   = 280
DROP_PATH_RATE = 0.1

# Порог уверенности: если p(со средником) >= порога — относим к классу
THRESHOLD    = 0.75   

DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}


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
    model_files = sorted(Path(models_dir).glob('fold_*_dinov2.pth'))
    if not model_files:
        raise FileNotFoundError(
            f"Модели не найдены в '{models_dir}'.\n"
            f"Убедитесь что папка models/ находится рядом со скриптом.")

    models = []
    for path in model_files:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    print(f"  Загружено моделей: {len(models)} (ансамбль)")
    return models


@torch.no_grad()
def predict(models, image_path, tta_transforms, device):
    """Возвращает вероятность класса 'со средником' и общую уверенность."""
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
    except Exception as e:
        print(f"  ⚠ Не удалось открыть {image_path.name}: {e}")
        return None, None

    all_probs = []
    for model in models:
        for tfm in tta_transforms:
            tensor = tfm(image=img)['image'].unsqueeze(0).to(device)
            logits = model(tensor).cpu()
            all_probs.append(F.softmax(logits, dim=1).squeeze(0))

    avg = torch.stack(all_probs).mean(0)
    p_srednik   = avg[1].item()
    confidence  = avg.max().item()
    return p_srednik, confidence


def main():
    parser = argparse.ArgumentParser(description='Инференс: сортировка переплётов')
    parser.add_argument('--input',     default=INPUT_DIR,  help='Папка с фото')
    parser.add_argument('--output',    default=OUTPUT_DIR, help='Папка для результатов')
    parser.add_argument('--models',    default=MODELS_DIR, help='Папка с моделями')
    parser.add_argument('--threshold', default=THRESHOLD,  type=float,
                        help='Порог уверенности (0.5–0.95, по умолчанию 0.75)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    print(f"Порог уверенности: {args.threshold:.0%}")
    print(f"Входная папка: {args.input}")

    # Папки результатов
    dir_yes    = os.path.join(args.output, 'со_средником')
    dir_no     = os.path.join(args.output, 'без_средника')
    dir_doubt  = os.path.join(args.output, 'сомнительные')
    for d in [dir_yes, dir_no, dir_doubt]:
        os.makedirs(d, exist_ok=True)

    # Загружаем модели и трансформы
    print("\nЗагружаем модели...")
    models      = load_models(args.models, device)
    tta_tfms    = get_tta_transforms()

    # Собираем список файлов
    image_files = sorted([
        p for p in Path(args.input).iterdir()
        if p.suffix.lower() in EXTENSIONS
    ])
    if not image_files:
        print(f"⚠ В папке '{args.input}' не найдено изображений.")
        return

    print(f"Найдено изображений: {len(image_files)}\n")

    # Прогоняем
    counts = {'со_средником': 0, 'без_средника': 0, 'сомнительные': 0, 'ошибок': 0}
    log_lines = ['filename,решение,уверенность,p_со_средником']

    for i, path in enumerate(image_files):
        p_srednik, confidence = predict(models, path, tta_tfms, device)

        if p_srednik is None:
            counts['ошибок'] += 1
            continue

        # Решение по порогу
        if p_srednik >= args.threshold:
            decision = 'со_средником'
            dst_dir  = dir_yes
        elif (1 - p_srednik) >= args.threshold:
            decision = 'без_средника'
            dst_dir  = dir_no
        else:
            decision = 'сомнительные'
            dst_dir  = dir_doubt

        counts[decision] += 1

        # Имя файла: добавляем уверенность спереди для удобства сортировки
        dst_name = f"conf{confidence:.2f}__{path.name}"
        shutil.copy2(path, os.path.join(dst_dir, dst_name))
        log_lines.append(f"{path.name},{decision},{confidence:.4f},{p_srednik:.4f}")

        # Прогресс
        if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
            print(f"  [{i+1}/{len(image_files)}] "
                  f"со средником: {counts['со_средником']} | "
                  f"без средника: {counts['без_средника']} | "
                  f"сомнительные: {counts['сомнительные']}")

    # Итоги
    total = len(image_files) - counts['ошибок']
    print(f"\n{'='*55}")
    print(f"  ИТОГ ({total} изображений, порог {args.threshold:.0%})")
    print(f"{'='*55}")
    print(f"  Со средником:  {counts['со_средником']:>4}  ({counts['со_средником']/total:.1%})")
    print(f"  Без средника:  {counts['без_средника']:>4}  ({counts['без_средника']/total:.1%})")
    print(f"  Сомнительные:  {counts['сомнительные']:>4}  ({counts['сомнительные']/total:.1%})")
    if counts['ошибок']:
        print(f"  Ошибок чтения: {counts['ошибок']:>4}")
    print(f"\n  Результаты в папке: {args.output}/")
    print(f"  Сомнительные ({counts['сомнительные']} шт.) стоит проверить вручную.")

    # Сохраняем лог
    log_path = os.path.join(args.output, 'inference_log.csv')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"  Лог: {log_path}")


if __name__ == '__main__':
    main()
