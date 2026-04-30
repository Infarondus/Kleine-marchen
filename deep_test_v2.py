"""
Трёхфазное обучение: голова → бэкбон постепенно → всё вместе
===============================================================

Фаза 1 (288px): только голова                    — быстрый старт
Фаза 2 (288px): последние 2 блока, голова FREEZE — бэкбон адаптируется
Фаза 3 (352px): последние 3 блока + голова       — fine-tuning всего

Ключевое правило: голова, которая хорошо обучилась в фазе 1,
НЕ трогается до фазы 3, и в фазе 3 обучается с очень малым LR.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# ═══════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════
WITH_SREDNIK_DIR    = './binding'
WITHOUT_SREDNIK_DIR = './not_binding'

BATCH_SIZE      = 16
SEED            = 42
NUM_FOLDS       = 5
NUM_WORKERS     = 4

LABEL_SMOOTHING = 0.0
DROP_RATE       = 0.3
DROP_PATH_RATE  = 0.1

# Размеры
PHASE1_SIZE     = 288
PHASE2_SIZE     = 288   # Фаза 2 на том же размере!
PHASE3_SIZE     = 352   # Только фаза 3 на большем

# Фаза 1: голова
PHASE1_EPOCHS   = 10
PHASE1_LR       = 1e-4
PHASE1_WD       = 1e-2
PHASE1_WARMUP   = 2

# Фаза 2: последние блоки, ГОЛОВА ЗАМОРОЖЕНА
PHASE2_EPOCHS   = 8
PHASE2_LR_BACK  = 1e-5   # Консервативно
PHASE2_WD       = 5e-3

# Фаза 3: fine-tuning всего
PHASE3_EPOCHS   = 15
PHASE3_LR_BACK  = 5e-6
PHASE3_LR_HEAD  = 1e-5   # Очень маленький — не ломаем обученную голову
PHASE3_WD       = 5e-3
PHASE3_LS       = 0.05

# EMA
EMA_DECAY       = 0.999

# Mixup/CutMix
MIXUP_ALPHA     = 0.2
CUTMIX_ALPHA    = 0.8
MIXUP_PROB      = 0.5

# Early stopping
PATIENCE        = 5
GRAD_CLIP       = 1.0

# Модель
MODEL_NAME      = 'convnext_tiny.fb_in22k'
# ═══════════════════════════════════════════════════


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(image_size: int, augment: bool, is_tta=False) -> A.Compose:
    if is_tta:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, value=[255, 255, 255]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    if augment:
        return A.Compose([
            A.LongestMaxSize(max_size=int(image_size * 1.25)),
            A.RandomResizedCrop(size=(image_size, image_size),
                                scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.15),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15,
                               rotate_limit=20, border_mode=0,
                               value=[255, 255, 255], p=0.6),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0),
            ], p=0.4),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.CoarseDropout(max_holes=4, max_height=image_size//8, max_width=image_size//8,
                            min_holes=1, min_height=image_size//20, min_width=image_size//20,
                            fill_value=128, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, value=[255, 255, 255]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_tta_transforms(image_size: int) -> list:
    base_before = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, value=[255, 255, 255]),
    ]
    base_after = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return [
        A.Compose(base_before + base_after),
        A.Compose(base_before + [A.HorizontalFlip(p=1.0)] + base_after),
        A.Compose(base_before + [A.VerticalFlip(p=1.0)] + base_after),
        A.Compose(base_before + [A.ShiftScaleRotate(
            shift_limit=0, scale_limit=0, rotate_limit=5, p=1.0)] + base_after),
        A.Compose(base_before + [A.HorizontalFlip(p=1.0), A.ShiftScaleRotate(
            shift_limit=0, scale_limit=0, rotate_limit=5, p=1.0)] + base_after),
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


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    _, _, h, w = x.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    return x, y, y[index], lam


def build_model(num_classes: int = 2) -> nn.Module:
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=DROP_RATE,
        drop_path_rate=DROP_PATH_RATE,
    )
    return model


def freeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def freeze_backbone(model: nn.Module):
    """Только голова обучается"""
    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ['head.fc', 'classifier', 'head'])
    _print_trainable(model)


def unfreeze_last_two_stages(model: nn.Module, freeze_head: bool = True):
    """Только последние 2 ConvNeXt stages, голова опционально freeze"""
    for name, param in model.named_parameters():
        if 'stages.2' in name or 'stages.3' in name:
            param.requires_grad = True
        elif not freeze_head and any(k in name for k in ['head.fc', 'classifier', 'head']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    _print_trainable(model)


def unfreeze_last_three_stages_and_head(model: nn.Module):
    """Последние 3 stages + голова (для фазы 3)"""
    for name, param in model.named_parameters():
        if 'stages.1' in name or 'stages.2' in name or 'stages.3' in name:
            param.requires_grad = True
        elif any(k in name for k in ['head.fc', 'classifier', 'head']):
            param.requires_grad = True
        else:
            param.requires_grad = False
    _print_trainable(model)


def _print_trainable(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Обучаемых: {trainable:,} / {total:,} ({trainable/total:.1%})")


class EMA:
    def __init__(self, model, decay=0.999):
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
                    self.shadow[k] = decay * self.shadow[k] + (1 - decay) * v.clone().detach()

    def apply_shadow(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


def run_epoch(model, loader, criterion, optimizer, scaler, device,
              scheduler, train: bool, use_mixup=False):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    nan_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            if train and use_mixup:
                r = np.random.rand()
                if r < MIXUP_PROB / 2:
                    images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                elif r < MIXUP_PROB:
                    images, labels_a, labels_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                else:
                    labels_a = labels_b = labels
                    lam = 1.0
            else:
                labels_a = labels_b = labels
                lam = 1.0

            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

            # NaN-защита
            if not torch.isfinite(loss):
                nan_batches += 1
                continue

            if train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
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
    avg_loss = total_loss / total if total > 0 else float('inf')
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate_tta(model, val_items: list, tta_tfms: list, device, batch_size: int):
    model.eval()
    all_probs = []

    for tfm in tta_tfms:
        ds = BindingDataset(val_items, tfm)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
        probs = []
        for images, _ in loader:
            images = images.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images).cpu()
                probs.append(F.softmax(logits, dim=1))
        all_probs.append(torch.cat(probs))

    avg_probs = torch.stack(all_probs).mean(0)
    true = torch.tensor([lbl for _, lbl in val_items])
    preds = avg_probs.argmax(dim=1).numpy()
    true_np = true.numpy()

    acc = accuracy_score(true_np, preds)
    f1 = f1_score(true_np, preds, average='macro', zero_division=0)
    return acc, f1, torch.log(avg_probs + 1e-8)


def make_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    """Warmup + cosine annealing"""
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_fold(fold_idx, train_items, val_items, device, use_amp):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"  Train: {len(train_items)} | Val: {len(val_items)}")
    print(f"  Val classes: {Counter(lbl for _, lbl in val_items)}")
    print(f"{'='*70}")

    # Weighted sampler
    train_labels = [lbl for _, lbl in train_items]
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    best_f1 = 0.0
    best_state = None
    ema = None

    # ═══════════════════════════════════════════
    # ФАЗА 1: только голова (288px)
    # ═══════════════════════════════════════════
    print(f"\n--- Фаза 1: {PHASE1_SIZE}px, только голова ---")
    train_ds = BindingDataset(train_items, get_transforms(PHASE1_SIZE, augment=True))
    val_ds = BindingDataset(val_items, get_transforms(PHASE1_SIZE, augment=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    freeze_backbone(model)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=PHASE1_LR, weight_decay=PHASE1_WD, eps=1e-8
    )
    scheduler = make_scheduler(opt1, PHASE1_EPOCHS, PHASE1_WARMUP, len(train_loader))
    ema = EMA(model, decay=EMA_DECAY)
    no_improve = 0

    for ep in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        tl, ta, tf1 = run_epoch(model, train_loader, criterion, opt1, scaler,
                                device, scheduler, train=True)
        vl, va, vf1 = run_epoch(model, val_loader, criterion, None, None,
                                device, None, train=False)
        ema.update(model)
        lr = opt1.param_groups[0]['lr']
        print(f"  Ep{ep:>2}/{PHASE1_EPOCHS} | lr:{lr:.2e} | loss:{tl:.4f}/{vl:.4f} | "
              f"acc:{ta:.1%}/{va:.1%} | F1:{tf1:.3f}/{vf1:.3f} | {time.time()-t0:.1f}s")

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

    # ═══════════════════════════════════════════
    # ФАЗА 2: последние 2 stages, ГОЛОВА FREEZE (288px)
    # ═══════════════════════════════════════════
    print(f"\n--- Фаза 2: {PHASE2_SIZE}px, stages 2-3, голова FREEZE ---")
    if best_state is not None:
        model.load_state_dict(best_state)

    unfreeze_last_two_stages(model, freeze_head=True)

    # Оптимизатор ТОЛЬКО для бэкбона
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    opt2 = torch.optim.AdamW(backbone_params, lr=PHASE2_LR_BACK,
                             weight_decay=PHASE2_WD, eps=1e-8)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PHASE2_EPOCHS, eta_min=1e-7)
    ema = EMA(model, decay=EMA_DECAY)
    no_improve = 0

    for ep in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        tl, ta, tf1 = run_epoch(model, train_loader, criterion, opt2, scaler,
                                device, None, train=True)
        vl, va, vf1 = run_epoch(model, val_loader, criterion, None, None,
                                device, None, train=False)
        ema.update(model)
        sch2.step()

        mk = " <- BEST" if vf1 > best_f1 else ""
        print(f"  Ep{PHASE1_EPOCHS+ep:>2}/{PHASE1_EPOCHS+PHASE2_EPOCHS+PHASE3_EPOCHS} | "
              f"lr:{opt2.param_groups[0]['lr']:.2e} | loss:{tl:.4f}/{vl:.4f} | "
              f"acc:{ta:.1%}/{va:.1%} | F1:{tf1:.3f}/{vf1:.3f} | {time.time()-t0:.1f}s{mk}")

        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping фазы 2")
                break

    # ═══════════════════════════════════════════
    # ФАЗА 3: последние 3 stages + голова, 352px
    # ═══════════════════════════════════════════
    print(f"\n--- Фаза 3: {PHASE3_SIZE}px, stages 1-3 + голова, fine-tuning ---")
    if best_state is not None:
        model.load_state_dict(best_state)

    unfreeze_last_three_stages_and_head(model)

    # Пересоздаём loaders на большем размере
    train_ds = BindingDataset(train_items, get_transforms(PHASE3_SIZE, augment=True))
    val_ds = BindingDataset(val_items, get_transforms(PHASE3_SIZE, augment=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    tta_tfms = get_tta_transforms(PHASE3_SIZE)

    # Раздельные LR: бэкбон очень маленький, голова тоже маленький
    backbone_p = [p for n, p in model.named_parameters()
                  if p.requires_grad and not any(x in n for x in ['head.fc', 'classifier', 'head'])]
    head_p = [p for n, p in model.named_parameters()
              if p.requires_grad and any(x in n for x in ['head.fc', 'classifier', 'head'])]

    opt3 = torch.optim.AdamW([
        {'params': backbone_p, 'lr': PHASE3_LR_BACK, 'weight_decay': PHASE3_WD},
        {'params': head_p, 'lr': PHASE3_LR_HEAD, 'weight_decay': PHASE3_WD},
    ], eps=1e-8)

    criterion = nn.CrossEntropyLoss(label_smoothing=PHASE3_LS)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=PHASE3_EPOCHS, eta_min=1e-7)
    ema = EMA(model, decay=EMA_DECAY)
    no_improve = 0
    mixup_start_ep = 5  # Mixup только после 5 эпох адаптации

    for ep in range(1, PHASE3_EPOCHS + 1):
        t0 = time.time()
        use_mixup = ep >= mixup_start_ep
        tl, ta, tf1 = run_epoch(model, train_loader, criterion, opt3, scaler,
                                device, None, train=True, use_mixup=use_mixup)
        vl, va, vf1 = run_epoch(model, val_loader, criterion, None, None,
                                device, None, train=False)
        ema.update(model)

        # EMA TTA
        ema.apply_shadow(model)
        tta_acc, tta_f1, _ = evaluate_tta(model, val_items, tta_tfms, device, BATCH_SIZE)
        ema.restore(model)

        sch3.step()
        lr_back = opt3.param_groups[0]['lr']
        lr_head = opt3.param_groups[1]['lr']

        mk = " <- BEST" if tta_f1 > best_f1 else ""
        total_ep = PHASE1_EPOCHS + PHASE2_EPOCHS + ep
        print(f"  Ep{total_ep:>2}/{PHASE1_EPOCHS+PHASE2_EPOCHS+PHASE3_EPOCHS} | "
              f"lr_b/h:{lr_back:.2e}/{lr_head:.2e} | loss:{tl:.4f}/{vl:.4f} | "
              f"acc:{ta:.1%}/{va:.1%}/{tta_acc:.1%} | "
              f"F1:{tf1:.3f}/{vf1:.3f}/{tta_f1:.3f} | {time.time()-t0:.1f}s{mk}")

        if tta_f1 > best_f1:
            best_f1 = tta_f1
            ema.apply_shadow(model)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ema.restore(model)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping по F1 (patience={PATIENCE})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_acc, final_f1, logits = evaluate_tta(model, val_items, tta_tfms, device, BATCH_SIZE)

    print(f"\n  Fold {fold_idx+1} — TTA Acc: {final_acc:.2%} | TTA F1: {final_f1:.4f}")
    return model, final_acc, final_f1, logits


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)} | "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Модель: {MODEL_NAME}")
    print(f"Фазы: {PHASE1_EPOCHS}+{PHASE2_EPOCHS}+{PHASE3_EPOCHS} эпох")
    print(f"Размеры: {PHASE1_SIZE}→{PHASE2_SIZE}→{PHASE3_SIZE}px")

    items_pos = BindingDataset.collect_items(WITH_SREDNIK_DIR, label=1)
    items_neg = BindingDataset.collect_items(WITHOUT_SREDNIK_DIR, label=0)
    print(f"\nИзображений: с переплётом={len(items_pos)}, без={len(items_neg)}")
    if not items_pos or not items_neg:
        print("ОШИБКА: одна из папок пуста!")
        return

    all_items = items_pos + items_neg
    all_labels = [lbl for _, lbl in all_items]
    print(f"Всего: {len(all_items)} | Классы: {Counter(all_labels)}")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    oof_logits = {}
    os.makedirs('models', exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        train_items = [all_items[i] for i in train_idx]
        val_items = [all_items[i] for i in val_idx]

        model, fold_acc, fold_f1, fold_logits = train_fold(
            fold_idx, train_items, val_items, device, use_amp)
        fold_results.append((fold_acc, fold_f1))

        for i, idx in enumerate(val_idx):
            oof_logits[idx] = fold_logits[i]

        torch.save({
            'fold': fold_idx,
            'model_state_dict': model.state_dict(),
            'tta_acc': fold_acc,
            'tta_f1': fold_f1,
            'image_size': PHASE3_SIZE,
            'model_name': MODEL_NAME,
        }, f'models/fold_{fold_idx+1}_best.pth')

    print(f"\n{'='*70}")
    print(f"  ИТОГИ K-FOLD")
    print(f"{'='*70}")
    for i, (acc, f1) in enumerate(fold_results):
        print(f"  Fold {i+1}: Acc={acc:.2%} | F1={f1:.4f}")
    accs = [a for a, _ in fold_results]
    f1s = [f for _, f in fold_results]
    print(f"\n  Среднее: Acc={np.mean(accs):.2%} (+/-{np.std(accs):.2%}) | F1={np.mean(f1s):.4f}")

    if len(oof_logits) == len(all_items):
        all_indices = sorted(oof_logits.keys())
        ensemble_probs = torch.stack([F.softmax(oof_logits[i], dim=0) for i in all_indices]).mean(0)
        true_labels = [all_labels[i] for i in all_indices]
        preds = ensemble_probs.argmax(dim=1).numpy()
        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds, average='macro', zero_division=0)
        print(f"\n  ENSEMBLE: Acc={acc:.2%} | F1={f1:.4f}")

    print(f"\n  Лучший фолд: Acc={max(accs):.2%}")


if __name__ == '__main__':
    main()
