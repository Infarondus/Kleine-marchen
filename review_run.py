"""
Ручная проверка прогона скрапера + сверка с решениями модели.

1) Просмотр (окно tkinter, OpenCV не нужен):
       python review_run.py ./run_fond_228
   Снимки из трёх папок показываются по одному, вперемешку, без названия папки.
   Клавиши:
     1 — со средником   (центр декоративной композиции крышки)
     2 — без средника   (переплёт без средника; одиночная накладка на гладкой крышке — сюда)
     3 — не переплёт    (страница, папка, футляр и т. п.)
     4 — сомнительно    (не гадайте — ставьте 4)
     Backspace — назад к предыдущему снимку, Q / Esc — выход (прогресс сохраняется)
   Оценки пишутся в review.csv; при повторном запуске продолжает с места остановки.
   Когда все снимки оценены, отчёт строится автоматически.

2) Только отчёт:
       python review_run.py ./run_fond_228 --report
   Сверяет review.csv с log.csv (оценки модели), печатает сводку и пишет
   review_report.csv — по строке на снимок, с вашей меткой и уверенностью модели.
"""
import argparse
import csv
import math
import os
import random
import re
from collections import Counter

FOLDERS = {'1_со_средником': 'со_средником',
           '2_отклонено_по_среднику': 'без_средника',
           '3_переплёт_не_найден': 'не_переплёт'}
KEYS = {'1': 'со_средником', '2': 'без_средника', '3': 'не_переплёт', '4': 'сомнительно'}
TAU1, TAU2 = 0.60, 0.55


# ───────────────────────── общее ─────────────────────────

def list_items(run_dir):
    items = []
    for folder in FOLDERS:
        d = os.path.join(run_dir, folder)
        if os.path.isdir(d):
            items += [(folder, f) for f in sorted(os.listdir(d))
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return items


def load_review(run_dir):
    path = os.path.join(run_dir, 'review.csv')
    done = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                done[(r['folder'], r['file'])] = r['label']
    return done


def save_review(run_dir, done):
    with open(os.path.join(run_dir, 'review.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['folder', 'file', 'label'])
        for (folder, fname), lab in done.items():
            w.writerow([folder, fname, lab])


def manuscript_of(fname):
    """'s0.43__ОР_РГБ_228_15.jpg' -> '15';  'ОР_РГБ_122_52-3.jpg' -> '52-3'"""
    stem = os.path.splitext(fname.split('__')[-1])[0]
    m = re.match(r'ОР_РГБ_[^_]+_(.+)$', stem)
    return (m.group(1) if m else stem).lower()


# ───────────────────────── просмотр ─────────────────────────

def review(run_dir):
    import tkinter as tk
    from PIL import Image, ImageTk

    items = list_items(run_dir)
    random.Random(0).shuffle(items)
    done = load_review(run_dir)
    queue = [it for it in items if it not in done]
    print(f"Снимков: {len(items)}, уже оценено: {len(done)}, осталось: {len(queue)}")
    if not queue:
        return True

    st = {'i': 0, 'hist': []}
    root = tk.Tk()
    root.title('Проверка прогона')
    root.configure(bg='black')
    pic = tk.Label(root, bg='black')
    pic.pack()
    info = tk.Label(root, font=('Arial', 13), fg='white', bg='black')
    info.pack(pady=6)
    max_w = root.winfo_screenwidth() - 80
    max_h = root.winfo_screenheight() - 160

    def show():
        if st['i'] >= len(queue):
            root.destroy()
            return
        folder, fname = queue[st['i']]
        img = Image.open(os.path.join(run_dir, folder, fname))
        img.draft('RGB', (max_w, max_h))  # быстрое уменьшение больших JPEG
        img = img.convert('RGB')
        img.thumbnail((max_w, max_h))
        photo = ImageTk.PhotoImage(img)
        pic.configure(image=photo)
        pic.image = photo
        info.configure(text=f"осталось {len(items) - len(done)}   |   1 со средником   "
                            f"2 без средника   3 не переплёт   4 сомнительно   |   "
                            f"Backspace назад   Q выход")

    def on_key(e):
        k = e.keysym
        if k in KEYS:
            done[queue[st['i']]] = KEYS[k]
            save_review(run_dir, done)
            st['hist'].append(st['i'])
            st['i'] += 1
            show()
        elif k == 'BackSpace' and st['hist']:
            st['i'] = st['hist'].pop()
            done.pop(queue[st['i']], None)
            save_review(run_dir, done)
            show()
        elif k in ('q', 'Q', 'Escape'):
            root.destroy()

    root.bind('<Key>', on_key)
    show()
    root.focus_force()
    root.mainloop()
    print(f"Оценено: {len(done)} из {len(items)}")
    return len(done) == len(items)


# ───────────────────────── отчёт ─────────────────────────

def auc(pos, neg):
    """Вероятность, что случайный «со средником» получит оценку выше случайного «без»."""
    if not pos or not neg:
        return float('nan')
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fnum(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def report(run_dir):
    done = load_review(run_dir)
    if not done:
        print("review.csv пуст — сначала оцените снимки.")
        return
    log = {}
    with open(os.path.join(run_dir, 'log.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            log[str(r['manuscript']).lower()] = r

    rows = []
    for (folder, fname), human in done.items():
        lr = log.get(manuscript_of(fname), {})
        rows.append({'file': fname, 'manuscript': manuscript_of(fname),
                     'model': FOLDERS[folder], 'human': human,
                     'p_binding': fnum(lr.get('best_binding')),
                     'p_srednik': fnum(lr.get('srednik'))})

    print("\n" + "=" * 68)
    print(f"  СВЕРКА: ваша оценка × решение модели   (снимков: {len(rows)})")
    print("=" * 68)
    tab = Counter((r['model'], r['human']) for r in rows)
    hum = ['со_средником', 'без_средника', 'не_переплёт', 'сомнительно']
    print("  " + "модель / вы".ljust(16) + ''.join(f"{h[:12]:>14}" for h in hum))
    for m in FOLDERS.values():
        print(f"  {m:<16}" + ''.join(f"{tab[(m, h)]:>14}" for h in hum))
    doubt = sum(r['human'] == 'сомнительно' for r in rows)
    print(f"\n  Сомнительные ({doubt}) ниже не учитываются.")
    R = [r for r in rows if r['human'] != 'сомнительно']

    # ── этап 2: только снимки, где модель нашла переплёт и вы согласны, что это переплёт
    s2 = [r for r in R if r['model'] != 'не_переплёт'
          and r['human'] in ('со_средником', 'без_средника') and r['p_srednik'] is not None]
    pos = [r['p_srednik'] for r in s2 if r['human'] == 'со_средником']
    neg = [r['p_srednik'] for r in s2 if r['human'] == 'без_средника']
    tp = sum(p >= TAU2 for p in pos); fn = len(pos) - tp
    fp = sum(n >= TAU2 for n in neg); tn = len(neg) - fp
    print("\n  ── ЭТАП 2 (DINOv2), переплёты, найденные ConvNeXt ──")
    print(f"  со средником: {len(pos)}, без средника: {len(neg)}")
    print(f"  при τ2 = {TAU2}: верно отобрано {tp}, ложно отобрано {fp}, "
          f"пропущено {fn}, верно отклонено {tn}")
    if tp + fp:
        print(f"  точность отбора (precision): {tp / (tp + fp):.3f}")
    if pos:
        print(f"  полнота (recall):            {tp / len(pos):.3f}")
    print(f"  AUC (насколько хорошо оценка упорядочивает классы): {auc(pos, neg):.3f}")

    def bins(vals, edges):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            out.append((lo, hi, sum(lo <= v < hi for v in vals)))
        return out

    misses = sorted(p for p in pos if p < TAU2)
    if misses:
        print(f"\n  Пропущенные средники ({len(misses)}): какую p(средник) им дала модель")
        for lo, hi, c in bins(misses, [0, 0.2, 0.3, 0.4, 0.5, TAU2]):
            print(f"    {lo:.2f}–{hi:.2f}: {c:>3}  {'█' * c}")
        neg_med = sorted(neg)[len(neg) // 2] if neg else float('nan')
        print(f"    для сравнения: медиана у настоящих «без средника» = {neg_med:.3f}")
    falses = sorted(n for n in neg if n >= TAU2)
    if falses:
        print(f"\n  Ложные средники ({len(falses)}): p = " + ', '.join(f'{v:.2f}' for v in falses))

    print("\n  Иллюстрация: что было бы при других порогах τ2 "
          "(подбирать порог по этому же фонду нельзя — это подгонка под тест)")
    print(f"    {'τ2':>5} {'отобрано':>9} {'precision':>10} {'recall':>8}")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel_p = sum(p >= t for p in pos); sel_n = sum(n >= t for n in neg)
        prec = sel_p / (sel_p + sel_n) if sel_p + sel_n else float('nan')
        rec = sel_p / len(pos) if pos else float('nan')
        mark = '  ← рабочий' if abs(t - TAU2) < 1e-9 else ''
        print(f"    {t:>5.2f} {sel_p + sel_n:>9} {prec:>10.3f} {rec:>8.3f}{mark}")

    # ── этап 1
    print("\n  ── ЭТАП 1 (ConvNeXt) ──")
    nb = [r for r in R if r['model'] == 'не_переплёт']
    missed_b = [r for r in nb if r['human'] in ('со_средником', 'без_средника')]
    print(f"  в «переплёт не найден»: {len(nb)}, из них на деле переплёты: {len(missed_b)} "
          f"(со средником: {sum(r['human'] == 'со_средником' for r in missed_b)})")
    for r in sorted(missed_b, key=lambda r: -(r['p_binding'] or 0)):
        print(f"    {r['file']:<40} p(переплёт) = {r['p_binding']}")
    fb = [r for r in R if r['model'] != 'не_переплёт' and r['human'] == 'не_переплёт']
    print(f"  принято за переплёт, но не переплёт: {len(fb)}")

    # ── сквозной итог
    allpos = sum(r['human'] == 'со_средником' for r in R)
    sel = [r for r in R if r['model'] == 'со_средником']
    good = sum(r['human'] == 'со_средником' for r in sel)
    print("\n  ── СКВОЗНОЙ ИТОГ ПАЙПЛАЙНА ──")
    if sel:
        print(f"  в итоговой папке {len(sel)} снимков, из них со средником {good}: "
              f"точность {good / len(sel):.3f}")
    if allpos:
        print(f"  всего средников среди просмотренных: {allpos}, найдено {good}: "
              f"полнота {good / allpos:.3f}")
    print("  (полнота — по рукописям, где снимок вообще был получен; "
          "рукописи без изображений не учитываются)")

    out = os.path.join(run_dir, 'review_report.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'manuscript', 'model', 'human',
                                          'match', 'p_binding', 'p_srednik'])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r['model'] == r['human'], r['file'])):
            w.writerow({**r, 'match': 'да' if r['model'] == r['human'] else 'НЕТ'})
    print(f"\n  По каждому снимку: {out}  (расхождения — в начале файла)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', nargs='?', default=None)
    ap.add_argument('--report', action='store_true', help='только отчёт, без просмотра')
    a = ap.parse_args()

    # Ищем папку прогона: указанную, текущую, папку со скриптом или ./run_fond_228
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = ([a.run_dir] if a.run_dir else []) + ['.', here, './run_fond_228',
                                                      os.path.join(here, 'run_fond_228')]
    found = next((c for c in candidates
                  if any(os.path.isdir(os.path.join(c, f)) for f in FOLDERS)), None)
    if not found:
        print("Не нашёл папки 1_со_средником / 2_отклонено_по_среднику / 3_переплёт_не_найден.")
        print("Искал в: " + ', '.join(os.path.abspath(c) for c in candidates))
        raise SystemExit(1)
    a.run_dir = found
    print(f"Папка прогона: {os.path.abspath(found)}")
    if a.report or review(a.run_dir):
        report(a.run_dir)
