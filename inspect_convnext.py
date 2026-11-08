"""
Что за модели лежат в папке models/ у ConvNeXt и как их обучали.

Для каждого .pth печатает сохранённые метаданные (фолд, размер, точность и т.п.)
и проверяет главное: менялся ли backbone при обучении. Для этого веса сравниваются
с исходной предобученной convnext_tiny.fb_in22k. Если backbone совпадает с
исходным, а отличается только голова, то это linear probing.

Запуск (в папке ConvNeXt):  python inspect_convnext.py
Нужен интернет один раз: timm скачает исходные веса (~110 МБ).
Ничего не изменяет.
"""
import datetime
import os
from pathlib import Path

import torch
import timm

MODEL_NAME = 'convnext_tiny.fb_in22k'


def main():
    ref = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2).state_dict()
    files = sorted(Path('.').rglob('*.pth'))
    print(f"Найдено .pth: {len(files)}\n")
    fingerprints = {}
    for p in files:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M')
        print("=" * 70)
        print(f"{p}  ({p.stat().st_size / 1e6:.0f} МБ, изменён {mtime})")
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"  не открылся: {e}")
            continue
        sd = ck.get('model_state_dict', ck) if isinstance(ck, dict) else ck
        if isinstance(ck, dict):
            meta = {k: v for k, v in ck.items()
                    if not isinstance(v, (dict, torch.Tensor)) and k != 'model_state_dict'}
            print(f"  метаданные: {meta if meta else '(нет)'}")
        if not isinstance(sd, dict):
            print("  внутри не state_dict")
            continue
        keys = list(sd.keys())
        if any(k.startswith('module.') for k in keys):  # SWA/DataParallel
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
            keys = list(sd.keys())
        is_convnext = any(k.startswith('stages.') for k in keys) and 'stem.0.weight' in sd
        print(f"  архитектура: {'ConvNeXt' if is_convnext else 'НЕ ConvNeXt (' + keys[0] + ' ...)'}"
              f" | тензоров: {len(keys)}")
        if not is_convnext:
            continue

        changed = {'stem': 0, 'stages.0': 0, 'stages.1': 0, 'stages.2': 0,
                   'stages.3': 0, 'head': 0, 'прочее': 0}
        total = dict.fromkeys(changed, 0)
        for k, v in sd.items():
            if k not in ref or ref[k].shape != v.shape:
                continue
            part = next((g for g in changed if k.startswith(g)), 'прочее')
            total[part] += 1
            # допуск: усреднение весов (EMA/SWA) даёт ошибки округления ~1e-7
            if not torch.allclose(v.float(), ref[k].float(), rtol=1e-5, atol=1e-6):
                changed[part] += 1
        print("  изменено тензоров относительно предобученной модели:")
        for g in changed:
            if total[g]:
                print(f"    {g:<9}: {changed[g]:>3} из {total[g]}")
        backbone_changed = sum(changed[g] for g in changed if g not in ('head',))
        print(f"  => {'LINEAR PROBING (backbone не менялся)' if backbone_changed == 0 else 'backbone дообучался'}")
        if 'head.fc.weight' in sd:
            fp = tuple(round(x, 6) for x in sd['head.fc.weight'].float().flatten()[:16].tolist())
            fingerprints.setdefault(fp, []).append(str(p))

    print("\n" + "=" * 70)
    print("Одинаковые модели (совпадают веса головы):")
    same = [names for names in fingerprints.values() if len(names) > 1]
    for names in same:
        print("  " + "  ==  ".join(names))
    if not same:
        print("  совпадений нет — все модели разные")


if __name__ == '__main__':
    main()
