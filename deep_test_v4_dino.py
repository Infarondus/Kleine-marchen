"""
DINOv2 fine-tuning для задачи средник/без средника
====================================================

Почему DINOv2, а не ConvNeXt:
  - ConvNeXt обучен на ImageNet (естественные фото) → слабые фичи для
    структурных деталей переплётов (сканы, технические снимки)
  - DINOv2 обучен self-supervised на 142M изображений → универсальные
    фичи, которые хорошо захватывают текстуры и структурные паттерны
  - Attention-механизм ViT буквально «смотрит» на разные части изображения,
    что критично для поиска средника как локального элемента

Стратегия:
  Фаза 1 (HEAD_EPOCHS эп): только MLP-голова (линейный probe)
  Фаза 2 (TUNE_EPOCHS эп): голова + последние UNFREEZE_BLOCKS блоков
                             с дифференциальным LR

Модель: vit_small_patch14_dinov2 (21M параметров, быстро, точно)
Альтернатива: vit_base_patch14_dinov2 (86M, медленнее, точнее)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU!'}")
print(f"Версия PyTorch: {torch.__version__}")
 
# CONFIG
 
WITH_SREDNIK_DIR    = './selected'
WITHOUT_SREDNIK_DIR = './rejected'

# ── Модель ────────────────────────────────────────────
MODEL_NAME      = 'vit_small_patch14_dinov2.lvd142m'
# Если нужна выше точность (но медленнее ~3x):
# MODEL_NAME    = 'vit_base_patch14_dinov2.lvd142m'

IMAGE_SIZE      = 280       # кратно patch_size=14: 280=14*20. Оптимально для 6 GB VRAM
                            # Варианты: 224 (быстрее), 280 (баланс), 336 (нужно ~10+ GB)
BATCH_SIZE      = 32        # уменьшить до 8 если OOM

# ── Общее ─────────────────────────────────────────────
SEED            = 42
NUM_FOLDS       = 5
NUM_WORKERS     = 0
GRAD_CLIP       = 1.0
LABEL_SMOOTHING = 0.05

# ── Фаза 1: только голова ─────────────────────────────
HEAD_EPOCHS     = 6
LR_HEAD_P1      = 5e-4     # голова с нуля → нужен более высокий LR
WD_P1           = 1e-4

# ── Фаза 2: голова + часть блоков ────────────────────
TUNE_EPOCHS     = 20
UNFREEZE_BLOCKS = 4        # разморозить последние N блоков ViT (из 12 у small)
                            # 4 блока = ~треть модели
LR_HEAD_P2      = 3e-5
LR_BACKBONE_P2  = 2e-6     # в ~15 раз меньше головы
WD_P2           = 1e-2

#Регуляризация
DROP_RATE       = 0.3       # dropout в голове
DROP_PATH_RATE  = 0.1

#EMA + TTA
EMA_DECAY       = 0.9995
TTA_ENABLED     = True
PATIENCE        = 10        # early stopping в фазе 2
 


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#DINOv2 нормализация
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int, augment: bool) -> A.Compose:
    """ViT чувствителен к масштабу — не обрезаем агрессивно."""
    if augment:
        return A.Compose([
            A.LongestMaxSize(max_size=int(image_size * 1.15)),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, value=[255, 255, 255]),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            # Мягкие геометрические искажения
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, border_mode=0,
                               value=[255, 255, 255], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20,
                                     val_shift_limit=15, p=1.0),
            ], p=0.6),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.35),
            A.CLAHE(clip_limit=2.0, p=0.2),
            # CoarseDropout помогает модели не фиксироваться на одном месте
            A.CoarseDropout(max_holes=3,
                            max_height=image_size // 10, max_width=image_size // 10,
                            min_holes=1,
                            min_height=image_size // 20, min_width=image_size // 20,
                            fill_value=200, p=0.25),
            A.Normalize(mean=DINO_MEAN, std=DINO_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, value=[255, 255, 255]),
            A.Normalize(mean=DINO_MEAN, std=DINO_STD),
            ToTensorV2(),
        ])


def get_tta_transforms(image_size: int) -> list:
    base = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, value=[255, 255, 255]),
    ]
    norm = [A.Normalize(mean=DINO_MEAN, std=DINO_STD), ToTensorV2()]
    return [
        A.Compose(base + norm),
        A.Compose(base + [A.HorizontalFlip(p=1.0)] + norm),
        A.Compose(base + [A.VerticalFlip(p=1.0)] + norm),
        A.Compose([A.LongestMaxSize(max_size=int(image_size * 1.05)),
                   A.PadIfNeeded(min_height=image_size, min_width=image_size,
                                 border_mode=0, value=[255, 255, 255]),
                   A.CenterCrop(height=image_size, width=image_size)] + norm),
        A.Compose(base + [A.RandomBrightnessContrast(
                          brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + norm),
    ]


class BindingDataset(Dataset):
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}

    def __init__(self, items: list, transform: A.Compose):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(image=np.array(img))['image']
            return tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"⚠ Ошибка загрузки {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.items))

    @classmethod
    def collect_items(cls, directory: str, label: int) -> list:
        if not os.path.exists(directory):
            return []
        return [
            (os.path.join(directory, f), label)
            for f in sorted(os.listdir(directory))
            if os.path.splitext(f)[1].lower() in cls.EXTENSIONS
        ]


def build_model(num_classes: int = 2) -> nn.Module:
    """
    DINOv2 ViT-Small с кастомной головой.
    timm создаёт модель без финального классификатора (num_classes=0),
    потом навешиваем свою голову с dropout.
    """
    backbone = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,          # убираем стандартный head
        drop_path_rate=DROP_PATH_RATE,
        img_size=IMAGE_SIZE,    # переопределяем дефолтные 518px модели
        dynamic_img_size=True,  # позиционные эмбеддинги интерполируются под нужный размер
    )
    embed_dim = backbone.embed_dim  # 384 у small, 768 у base

    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 256),
        nn.GELU(),
        nn.Dropout(DROP_RATE),
        nn.Linear(256, num_classes),
    )

    # Инициализируем голову
    nn.init.xavier_uniform_(head[1].weight)
    nn.init.zeros_(head[1].bias)
    nn.init.xavier_uniform_(head[4].weight)
    nn.init.zeros_(head[4].bias)

    class DinoClassifier(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)   # [B, embed_dim] — CLS token
            return self.head(features)

    return DinoClassifier(backbone, head)


def set_phase1(model: nn.Module):
    """Замораживаем весь backbone, обучаем только голову."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    _print_trainable(model, "Фаза 1 (только голова)")


def set_phase2(model: nn.Module, n_blocks: int):
    """
    Размораживаем последние n_blocks трансформер-блоков и голову.
    В ViT блоки хранятся в model.backbone.blocks (список).
    """
    # Сначала всё замораживаем
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем голову
    for param in model.head.parameters():
        param.requires_grad = True

    # Размораживаем последние n_blocks блоков + norm
    total_blocks = len(model.backbone.blocks)
    unfreeze_from = total_blocks - n_blocks
    for i, block in enumerate(model.backbone.blocks):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    # Размораживаем финальную LayerNorm бэкбона
    if hasattr(model.backbone, 'norm'):
        for param in model.backbone.norm.parameters():
            param.requires_grad = True

    _print_trainable(model, f"Фаза 2 (голова + последние {n_blocks}/{total_blocks} блоков)")


def _print_trainable(model: nn.Module, label: str):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [{label}] Обучаемых: {trainable:,} / {total:,} ({trainable/total:.1%})")


def make_optimizer_p1(model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=LR_HEAD_P1, weight_decay=WD_P1, eps=1e-8)


def make_optimizer_p2(model: nn.Module):
    """Дифференциальный LR: голова быстрее, блоки бэкбона медленнее."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"  LR: голова={LR_HEAD_P2:.1e}, блоки бэкбона={LR_BACKBONE_P2:.1e} "
          f"(соотношение ×{LR_HEAD_P2/LR_BACKBONE_P2:.0f})")
    return torch.optim.AdamW([
        {'params': head_params,     'lr': LR_HEAD_P2,     'weight_decay': WD_P2},
        {'params': backbone_params, 'lr': LR_BACKBONE_P2, 'weight_decay': WD_P2},
    ], eps=1e-8)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int = 0):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.num_updates = 0
        self.backup = {}

    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k] = decay * self.shadow[k] + (1 - decay) * v.detach()

    def apply_shadow(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


def run_epoch(model, loader, criterion, optimizer, scaler, device, scheduler, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    nan_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast('cuda', enabled=False):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                nan_batches += 1
                if train and optimizer:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                    else:
                        nan_batches += 1
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if nan_batches > 0:
        print(f"    [NaN-батчей пропущено: {nan_batches}]")

    acc = correct / total if total > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) if all_labels else 0
    return total_loss / total if total > 0 else float('inf'), acc, f1


@torch.no_grad()
def evaluate_tta(model, val_items, tta_tfms, device, batch_size):
    model.eval()
    all_probs = []
    for tfm in tta_tfms:
        ds = BindingDataset(val_items, tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)
        probs = []
        for images, _ in loader:
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images.to(device)).cpu()
            probs.append(F.softmax(logits, dim=1))
        all_probs.append(torch.cat(probs))

    avg_probs = torch.stack(all_probs).mean(0)
    true_np = np.array([lbl for _, lbl in val_items])
    preds = avg_probs.argmax(dim=1).numpy()
    return (accuracy_score(true_np, preds),
            f1_score(true_np, preds, average='macro', zero_division=0),
            torch.log(avg_probs + 1e-8))


def train_fold(fold_idx, train_items, val_items, device, use_amp):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"  Train: {len(train_items)} | Val: {len(val_items)}")
    print(f"  Val классы: {Counter(lbl for _, lbl in val_items)}")
    print(f"{'='*70}")

    train_ds = BindingDataset(train_items, get_transforms(IMAGE_SIZE, augment=True))
    val_ds   = BindingDataset(val_items,   get_transforms(IMAGE_SIZE, augment=False))

    # Взвешенный семплер для баланса классов
    train_labels  = [lbl for _, lbl in train_items]
    class_counts  = Counter(train_labels)
    sample_weights = [1.0 / class_counts[lbl] for lbl in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=False)

    model     = build_model(num_classes=2).to(device)
    dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Тест forward pass: input={dummy.device}, output={out.device}, shape={out.shape}")
    print(f"  VRAM после forward: {torch.cuda.memory_allocated()/1e6:.0f} MB")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler    = torch.cuda.amp.GradScaler() if use_amp else None
    tta_tfms  = get_tta_transforms(IMAGE_SIZE)
    ema       = EMA(model, decay=EMA_DECAY)

    best_f1    = 0.0
    best_state = None
    

    # ФАЗА 1: только голова
    print(f"\n  ── ФАЗА 1: только голова ({HEAD_EPOCHS} эп, LR={LR_HEAD_P1:.1e}) ──")
    set_phase1(model)
    optimizer_p1 = make_optimizer_p1(model)
    # Warmup на первые 2 эпохи, потом cosine decay
    p1_steps  = len(train_loader) * HEAD_EPOCHS
    scheduler_p1 = make_scheduler(optimizer_p1, p1_steps,
                                  warmup_steps=len(train_loader) * 2)

    for ep in range(1, HEAD_EPOCHS + 1):
        t0 = time.time()
        tl, ta, tf1 = run_epoch(model, train_loader, criterion, optimizer_p1,
                                scaler, device, scheduler_p1, train=True)
        vl, va, vf1 = run_epoch(model, val_loader, criterion, None,
                                None, device, None, train=False)
        ema.update(model)
        lr = optimizer_p1.param_groups[0]['lr']

        # Быстрая EMA-оценка
        ema.apply_shadow(model)
        ema_acc, ema_f1, _ = evaluate_tta(model, val_items,
                                          [tta_tfms[0]], device, BATCH_SIZE)
        ema.restore(model)

        mk = " <- BEST" if ema_f1 > best_f1 else ""
        print(f"  [P1] Ep{ep:>2}/{HEAD_EPOCHS} | lr:{lr:.2e} | "
              f"loss:{tl:.4f}/{vl:.4f} | acc:{ta:.1%}/{va:.1%} | "
              f"F1:{tf1:.3f}/{vf1:.3f} | EMA:{ema_acc:.1%}/{ema_f1:.3f} | "
              f"{time.time()-t0:.1f}s{mk}")

        if ema_f1 > best_f1:
            best_f1    = ema_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ФАЗА 2: голова + блоки бэкбона
    print(f"\n  ── ФАЗА 2: fine-tuning ({TUNE_EPOCHS} эп, "
          f"последние {UNFREEZE_BLOCKS} блоков) ──")

    # Загружаем лучшее состояние фазы 1
    if best_state:
        model.load_state_dict(best_state)

    set_phase2(model, UNFREEZE_BLOCKS)
    optimizer_p2 = make_optimizer_p2(model)
    p2_steps     = len(train_loader) * TUNE_EPOCHS
    # Небольшой warmup (1 эп)
    scheduler_p2 = make_scheduler(optimizer_p2, p2_steps,
                                  warmup_steps=len(train_loader))

    no_improve = 0
    for ep in range(1, TUNE_EPOCHS + 1):
        t0 = time.time()
        tl, ta, tf1 = run_epoch(model, train_loader, criterion, optimizer_p2,
                                scaler, device, scheduler_p2, train=True)
        vl, va, vf1 = run_epoch(model, val_loader, criterion, None,
                                None, device, None, train=False)
        ema.update(model)
        lr_h = optimizer_p2.param_groups[0]['lr']
        lr_b = optimizer_p2.param_groups[1]['lr']

        ema.apply_shadow(model)
        ema_acc, ema_f1, _ = evaluate_tta(model, val_items,
                                          [tta_tfms[0]], device, BATCH_SIZE)
        ema.restore(model)

        mk = " <- BEST" if ema_f1 > best_f1 else ""
        print(f"  [P2] Ep{ep:>2}/{TUNE_EPOCHS} | "
              f"lr(h):{lr_h:.1e} lr(b):{lr_b:.1e} | "
              f"loss:{tl:.4f}/{vl:.4f} | acc:{ta:.1%}/{va:.1%} | "
              f"F1:{tf1:.3f}/{vf1:.3f} | EMA:{ema_acc:.1%}/{ema_f1:.3f} | "
              f"{time.time()-t0:.1f}s{mk}")

        if ema_f1 > best_f1:
            best_f1    = ema_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping на эпохе {ep} (patience={PATIENCE})")
                break


    # Финальная оценка с полным TTA
    if best_state:
        model.load_state_dict(best_state)

    no_tta_acc, no_tta_f1, _      = evaluate_tta(model, val_items,
                                                  [tta_tfms[0]], device, BATCH_SIZE)
    tta_acc,    tta_f1,    logits = evaluate_tta(model, val_items,
                                                  tta_tfms,       device, BATCH_SIZE)

    print(f"\n  Fold {fold_idx+1} результат:")
    print(f"    No TTA: {no_tta_acc:.2%} / F1={no_tta_f1:.4f}")
    print(f"    TTA ×{len(tta_tfms)}: {tta_acc:.2%} / F1={tta_f1:.4f}")

    return model, best_state, no_tta_acc, tta_acc, tta_f1, logits


def main():
    set_seed(SEED)
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    print(f"\nМодель:   {MODEL_NAME}")
    print(f"Размер:   {IMAGE_SIZE}px | Батч: {BATCH_SIZE}")
    print(f"Стратегия: двухфазный fine-tuning")
    print(f"  Фаза 1: {HEAD_EPOCHS} эп, только голова, LR={LR_HEAD_P1:.1e}")
    print(f"  Фаза 2: {TUNE_EPOCHS} эп, последние {UNFREEZE_BLOCKS} блоков, "
          f"LR(head)={LR_HEAD_P2:.1e} / LR(backbone)={LR_BACKBONE_P2:.1e}")

    items_pos = BindingDataset.collect_items(WITH_SREDNIK_DIR, label=1)
    items_neg = BindingDataset.collect_items(WITHOUT_SREDNIK_DIR, label=0)
    print(f"\nДанные: с средником={len(items_pos)}, без средника={len(items_neg)}")
    if not items_pos or not items_neg:
        print("ОШИБКА: одна из папок пуста!")
        return

    all_items  = items_pos + items_neg
    all_labels = [lbl for _, lbl in all_items]
    print(f"Итого: {len(all_items)} | {Counter(all_labels)}")

    skf         = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    oof_logits   = {}
    os.makedirs('models', exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(all_labels)), all_labels)):
        train_items = [all_items[i] for i in train_idx]
        val_items   = [all_items[i] for i in val_idx]

        model, best_state, no_tta_acc, tta_acc, tta_f1, fold_logits = train_fold(
            fold_idx, train_items, val_items, device, use_amp)

        fold_results.append((no_tta_acc, tta_acc, tta_f1))
        for i, idx in enumerate(val_idx):
            oof_logits[idx] = fold_logits[i]

        torch.save({
            'fold':             fold_idx,
            'model_state_dict': best_state,
            'no_tta_acc':       no_tta_acc,
            'tta_acc':          tta_acc,
            'tta_f1':           tta_f1,
            'image_size':       IMAGE_SIZE,
            'model_name':       MODEL_NAME,
        }, f'models/fold_{fold_idx+1}_dinov2.pth')

    #итоги
    print(f"\n{'='*70}")
    print(f"  ИТОГИ ({NUM_FOLDS}-FOLD) — DINOv2 двухфазный fine-tuning")
    print(f"{'='*70}")
    for i, (no_tta, tta, f1) in enumerate(fold_results):
        print(f"  Fold {i+1}: No TTA={no_tta:.2%} | TTA={tta:.2%} | F1={f1:.4f}")

    no_ttas = [a for a, _, _ in fold_results]
    ttas    = [a for _, a, _ in fold_results]
    f1s     = [f for _, _, f in fold_results]
    print(f"\n  No TTA: {np.mean(no_ttas):.2%} ± {np.std(no_ttas):.2%}")
    print(f"  TTA:    {np.mean(ttas):.2%} ± {np.std(ttas):.2%}")
    print(f"  F1:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # OOF ensemble
    if len(oof_logits) == len(all_items):
        all_idx  = sorted(oof_logits.keys())
        probs    = torch.stack(
            [F.softmax(oof_logits[i], dim=0) for i in all_idx])
        true_np  = np.array([all_labels[i] for i in all_idx])
        preds    = probs.argmax(dim=1).numpy()
        acc  = accuracy_score(true_np, preds)
        f1   = f1_score(true_np, preds, average='macro', zero_division=0)
        prec = precision_score(true_np, preds, average='macro', zero_division=0)
        rec  = recall_score(true_np, preds, average='macro', zero_division=0)
        cm   = confusion_matrix(true_np, preds)
        print(f"\n  {'─'*50}")
        print(f"  OOF ENSEMBLE (все {NUM_FOLDS} фолдов):")
        print(f"    Accuracy:  {acc:.2%}")
        print(f"    F1-macro:  {f1:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
        print(f"\n    Confusion Matrix (строки=реальные, столбцы=предсказанные):")
        print(f"                   Нет средника  Есть средник")
        print(f"    Нет средника       {cm[0,0]:>4}          {cm[0,1]:>4}")
        print(f"    Есть средник       {cm[1,0]:>4}          {cm[1,1]:>4}")
        print(f"  {'─'*50}")

    print(f"\n  Модели: models/fold_*_dinov2.pth")


if __name__ == '__main__':
    main()
