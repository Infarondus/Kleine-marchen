"""
Честная out-of-fold (OOF) оценка ConvNeXt-Tiny (этап 1: переплёт / не-переплёт).

Обучение — тем же рецептом deep_test_v2.py, без изменений (функция train_fold
импортируется оттуда). Отличия только в разбиении данных:
  - точные дубли (одна рукопись дважды в одном классе) удаляются;
  - разбиение на 5 фолдов делается с группировкой по рукописи
    (StratifiedGroupKFold): переплёт и не-переплёт одной рукописи всегда
    попадают в один фолд, поэтому модель не видит «соседний» снимок той же книги.

Каждое изображение оценивается только моделью, которая его не видела:
  - как в боевом скрипте: одна трансформация, размер из чекпоинта (352 px);
  - с 5 TTA, как в deep_test_v2.py.

Запуск (в папке ConvNeXt, рядом с deep_test_v2.py, binding/, not_binding/):
    python convnext_oof.py              # все данные      -> models_oof_full/
    python convnext_oof.py --balanced   # классы поровну  -> models_oof_balanced/
Старые .pth не трогаются.
Результат: сводка в консоли + convnext_oof_full.csv / convnext_oof_balanced.csv.
"""
import argparse
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

import deep_test_v2 as dt  # тот же рецепт обучения, без изменений

THRESHOLD = 0.60  # τ1 боевого пайплайна


def manuscript_key(path):
    name = re.sub(r'^conf[\d.]+__', '', os.path.basename(path))
    return os.path.splitext(name.replace(' ', '_').lower())[0]


@torch.no_grad()
def predict_plain(model, items, size, device):
    """Как в scrape_bindings_v2.py: одна трансформация без TTA."""
    model.eval()
    ds = dt.BindingDataset(items, dt.get_transforms(size, augment=False))
    out = []
    for x, _ in DataLoader(ds, batch_size=dt.BATCH_SIZE, shuffle=False, num_workers=0):
        out.append(F.softmax(model(x.to(device)).float().cpu(), dim=1))
    return torch.cat(out)


def report(title, y, prob):
    p = prob.argmax(1).numpy()
    cm = confusion_matrix(y, p, labels=[0, 1])
    conf = prob.max(1).values.numpy()
    sure = conf >= THRESHOLD
    print(f"\n  {title} (n={len(y)})")
    print(f"    Accuracy {accuracy_score(y, p):.4f} | Precision "
          f"{precision_score(y, p, average='macro', zero_division=0):.4f} | Recall "
          f"{recall_score(y, p, average='macro', zero_division=0):.4f} | F1 "
          f"{f1_score(y, p, average='macro', zero_division=0):.4f}")
    print(f"    не-переплёт: {cm[0,0]} верно / {cm[0,1]} ошибочно "
          f"(полнота {cm[0,0]/cm[0].sum():.3f})")
    print(f"    переплёт:    {cm[1,1]} верно / {cm[1,0]} ошибочно "
          f"(полнота {cm[1,1]/cm[1].sum():.3f})")
    print(f"    уверенность < {THRESHOLD}: {(~sure).mean():.1%}; на уверенных: "
          f"Accuracy {accuracy_score(y[sure], p[sure]):.4f}, "
          f"F1 {f1_score(y[sure], p[sure], average='macro', zero_division=0):.4f}")
    pb = prob[:, 1].numpy() >= THRESHOLD
    print(f"    отбор p(переплёт) >= {THRESHOLD}: {pb.sum()} шт., precision "
          f"{(y[pb] == 1).mean():.4f}, recall {(pb & (y == 1)).sum() / (y == 1).sum():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--balanced', action='store_true',
                    help='оставить столько же переплётов, сколько не-переплётов (как в апреле)')
    args = ap.parse_args()
    mode = 'balanced' if args.balanced else 'full'
    out_dir = f'models_oof_{mode}'

    dt.set_seed(dt.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Устройство: {torch.cuda.get_device_name(0) if use_amp else 'ПРОЦЕССОР (CPU)'}")

    raw = (dt.BindingDataset.collect_items(dt.WITH_SREDNIK_DIR, 1) +
           dt.BindingDataset.collect_items(dt.WITHOUT_SREDNIK_DIR, 0))
    items, seen, dropped = [], set(), 0
    for path, label in raw:
        k = (manuscript_key(path), label)
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        items.append((path, label))
    print(f"Изображений: {len(raw)}, удалено дублей: {dropped}, осталось: {len(items)} "
          f"{Counter(l for _, l in items)}")

    # Режим balanced: случайно оставляем столько переплётов, сколько не-переплётов.
    # Отложенные переплёты не участвуют в обучении ни одного фолда и в конце
    # проверяются ансамблем всех 5 моделей (его не видела ни одна).
    extra = []
    if args.balanced:
        pos = [it for it in items if it[1] == 1]
        neg = [it for it in items if it[1] == 0]
        rng = np.random.RandomState(dt.SEED)
        keep = set(rng.choice(len(pos), size=len(neg), replace=False).tolist())
        extra = [pos[i] for i in range(len(pos)) if i not in keep]
        items = [pos[i] for i in sorted(keep)] + neg
        print(f"Режим balanced: в обучении/OOF {Counter(l for _, l in items)}, "
              f"отложено переплётов: {len(extra)}")
    else:
        print("Режим full: все данные")

    y = np.array([l for _, l in items])
    groups = [manuscript_key(p) for p, _ in items]
    print(f"Рукописей (групп): {len(set(groups))} | модели -> {out_dir}/")

    sgkf = StratifiedGroupKFold(n_splits=dt.NUM_FOLDS, shuffle=True, random_state=dt.SEED)
    prob_plain = torch.zeros(len(items), 2)
    prob_tta = torch.zeros(len(items), 2)
    prob_extra = torch.zeros(len(extra), 2)
    fold_of = np.zeros(len(items), dtype=int)
    os.makedirs(out_dir, exist_ok=True)
    ref = dt.build_model().state_dict()  # исходные предобученные веса для сравнения

    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(items)), y, groups)):
        assert not ({groups[i] for i in tr} & {groups[i] for i in va}), "утечка рукописи"
        train_items = [items[i] for i in tr]
        val_items = [items[i] for i in va]
        model, _, _, logp = dt.train_fold(k, train_items, val_items, device, use_amp)
        prob_tta[va] = logp.exp().float()
        prob_plain[va] = predict_plain(model, val_items, dt.PHASE3_SIZE, device)
        if extra:
            prob_extra += predict_plain(model, extra, dt.PHASE3_SIZE, device) / dt.NUM_FOLDS
        fold_of[va] = k + 1
        cur = model.state_dict()
        backbone_trained = any(
            not torch.allclose(cur[n].float().cpu(), ref[n].float(), rtol=1e-5, atol=1e-6)
            for n in ref if n.startswith('stages'))
        print(f"  fold {k+1}: backbone {'дообучен' if backbone_trained else 'НЕ менялся (linear probing)'}")
        torch.save({'fold': k, 'model_state_dict': model.state_dict(),
                    'image_size': dt.PHASE3_SIZE, 'model_name': dt.MODEL_NAME,
                    'backbone_trained': backbone_trained},
                   f'{out_dir}/fold_{k+1}_oof.pth')
        del model
        if use_amp:
            torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print(f"  ЧЕСТНАЯ OOF-ОЦЕНКА ConvNeXt, режим {mode}")
    print("  (каждое изображение — моделью, не видевшей его)")
    print("=" * 64)
    report(f"Как в боевом скрипте: без TTA, {dt.PHASE3_SIZE} px", y, prob_plain)
    report(f"С 5 TTA, {dt.PHASE3_SIZE} px", y, prob_tta)
    if extra:
        found = (prob_extra[:, 1] >= THRESHOLD).float().mean().item()
        print(f"\n  Отложенные переплёты (n={len(extra)}), ансамбль 5 моделей, ни одна их не видела:")
        print(f"    распознано как переплёт (p >= {THRESHOLD}): {found:.1%}")

    csv_name = f'convnext_oof_{mode}.csv'
    with open(csv_name, 'w', encoding='utf-8') as f:
        f.write('path,manuscript,fold,true,p_binding_plain,p_binding_tta\n')
        for i, (p, l) in enumerate(items):
            f.write(f"{p},{groups[i]},{fold_of[i]},{l},"
                    f"{float(prob_plain[i,1]):.4f},{float(prob_tta[i,1]):.4f}\n")
        for i, (p, l) in enumerate(extra):
            f.write(f"{p},{manuscript_key(p)},extra,{l},{float(prob_extra[i,1]):.4f},\n")
    print(f"\n  Подробно: {csv_name}")


if __name__ == '__main__':
    main()
