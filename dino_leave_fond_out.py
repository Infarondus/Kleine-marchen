"""
Проверка обобщения DINOv2 на НЕЗНАКОМЫЙ фонд (leave-one-fond-out).

Модель обучается тем же рецептом, что в deep_test_v4_dino.py, но на всех
фондах, КРОМЕ одного. Выбранный фонд модель не видит ни при обучении, ни при
выборе лучшей эпохи: валидация для ранней остановки берётся из обучающих
фондов (≈1/6, с группировкой по рукописи, чтобы дубли не разъезжались).
Затем модель оценивается на исключённом фонде с 5 TTA.

Запуск (в папке DINO, рядом с deep_test_v4_dino.py, selected/, rejected/):
    python dino_leave_fond_out.py --fond 98
    python dino_leave_fond_out.py --fond 304-i
Результат: сводка в консоли + lofo_<фонд>.csv. Модель сохраняется в
models_lofo/ (старые fold_*_km.pth не трогаются).
"""
import argparse
import os
import re
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)

import deep_test_v4_dino as dt  # тот же рецепт обучения, без изменений


def manuscript_key(path):
    name = re.sub(r'^conf[\d.]+__', '', os.path.basename(path))
    return os.path.splitext(name.replace(' ', '_').lower())[0]


def fond_of(path):
    m = re.match(r'ор_ргб_(.+)_(\d+)$', manuscript_key(path))
    return m.group(1) if m else '?'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fond', required=True, help='какой фонд исключить, напр. 98 или 304-i')
    args = ap.parse_args()
    hold = args.fond.lower()

    dt.set_seed(dt.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Устройство: {torch.cuda.get_device_name(0) if use_amp else 'ПРОЦЕССОР (CPU)'}")

    items = (dt.BindingDataset.collect_items(dt.WITH_SREDNIK_DIR, 1) +
             dt.BindingDataset.collect_items(dt.WITHOUT_SREDNIK_DIR, 0))
    test = [it for it in items if fond_of(it[0]) == hold]
    rest = [it for it in items if fond_of(it[0]) != hold]
    if not test:
        print(f"Фонд '{hold}' не найден. Есть: {sorted(Counter(fond_of(p) for p, _ in items))}")
        return

    # Внутренняя валидация (для ранней остановки) — только из обучающих фондов
    y = [l for _, l in rest]
    groups = [manuscript_key(p) for p, _ in rest]
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=dt.SEED)
    tr_idx, va_idx = next(sgkf.split(np.zeros(len(rest)), y, groups))
    train_items = [rest[i] for i in tr_idx]
    val_items = [rest[i] for i in va_idx]

    print(f"Исключённый фонд {hold}: {len(test)} изобр. {Counter(l for _, l in test)}")
    print(f"Обучение: {len(train_items)} | валидация (ранняя остановка): {len(val_items)}")

    model, best_state, *_ = dt.train_fold(0, train_items, val_items, device, use_amp)
    os.makedirs('models_lofo', exist_ok=True)
    torch.save({'model_state_dict': best_state, 'held_out_fond': hold,
                'image_size': dt.IMAGE_SIZE, 'model_name': dt.MODEL_NAME},
               f'models_lofo/lofo_{hold}.pth')

    model.load_state_dict(best_state)
    acc, f1, logp = dt.evaluate_tta(model, test, dt.get_tta_transforms(dt.IMAGE_SIZE),
                                    device, dt.BATCH_SIZE)
    probs = logp.exp()
    yt = np.array([l for _, l in test])
    yp = probs.argmax(1).numpy()
    cm = confusion_matrix(yt, yp, labels=[0, 1])

    print("\n" + "=" * 64)
    print(f"  НЕЗНАКОМЫЙ ФОНД {hold} (n={len(test)}), модель его не видела")
    print("=" * 64)
    print(f"  Accuracy {accuracy_score(yt, yp):.4f} | F1-macro "
          f"{f1_score(yt, yp, average='macro', zero_division=0):.4f} | Precision "
          f"{precision_score(yt, yp, average='macro', zero_division=0):.4f} | Recall "
          f"{recall_score(yt, yp, average='macro', zero_division=0):.4f}")
    print(f"  без средника: {cm[0,0]} верно / {cm[0,1]} ошибочно")
    print(f"  со средником: {cm[1,1]} верно / {cm[1,0]} ошибочно")

    with open(f'lofo_{hold}.csv', 'w', encoding='utf-8') as f:
        f.write('path,true,pred,p_srednik\n')
        for (p, l), pr in zip(test, probs):
            f.write(f"{p},{l},{int(pr.argmax())},{float(pr[1]):.4f}\n")
    print(f"\n  Подробно: lofo_{hold}.csv")


if __name__ == '__main__':
    main()
