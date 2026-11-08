"""
Двухступенчатый скрапер переплётов со средником — версия 3 (для независимой апробации)

Ступень 1 — ConvNeXt (ансамбль новых моделей из models_oof_full): находит переплёт
Ступень 2 — DINOv2 (ансамбль 5 фолдов + TTA): проверяет наличие средника

Отличия от v2:
  * ВСЕ модели лежат в одной папке MODELS_DIR; тип модели (ConvNeXt или DINOv2)
    определяется автоматически по содержимому файла, имена не важны.
  * ConvNeXt работает ансамблем всех найденных моделей, размер берётся из чекпоинта.
  * Ведётся журнал по КАЖДОЙ рукописи (log.csv): оценки всех кадров, выбранный кадр,
    оценка средника и решение. Это нужно, чтобы потом посчитать точность и полноту.
  * Сохраняются не только отобранные, но и отклонённые переплёты — в отдельные
    папки, чтобы их можно было просмотреть и найти пропуски:
        <SAVE_DIR>/1_со_средником/          — итог пайплайна
        <SAVE_DIR>/2_отклонено_по_среднику/ — переплёт найден, но p(средник) < τ2
        <SAVE_DIR>/3_переплёт_не_найден/    — лучший кадр рукописи, p(переплёт) < τ1
  * run_info.json — параметры запуска и контрольные суммы моделей (для статьи).
  * Можно прервать и перезапустить: уже обработанные рукописи (есть в log.csv)
    пропускаются.

Запуск: python scrape_bindings_v3.py
Рядом со скриптом: msedgedriver.exe и папка MODELS_DIR с 10 файлами .pth
(5 ConvNeXt из models_oof_full/ и 5 DINOv2 fold_*_km.pth).
"""

import csv
import datetime
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import unquote

import albumentations as A
import numpy as np
import requests
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from bs4 import BeautifulSoup
from PIL import Image

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# ═══════════════════════════ CONFIG ═══════════════════════════
COLLECTION_NAME     = "228"
BASE_COLLECTION_URL = f"https://lib-fond.ru/lib-rgb/{COLLECTION_NAME}"
START_PAGE          = 1
END_PAGE            = 12      # скрапер сам остановится на первой пустой странице

MODELS_DIR          = "./models_pipeline"     # ← сюда все 10 моделей
SAVE_DIR            = f"./run_fond_{COLLECTION_NAME}"

BINDING_MIN_SCORE   = 0.60      # τ1
SREDNIK_MIN_SCORE   = 0.55      # τ2
EARLY_STOP_CONFIDENCE = 0.95    # p_es

IMAGES_TO_CHECK     = 5
WAIT_TIME           = 20
REQUEST_DELAY       = 0.3
SEED                = 42

DINO_MODEL_NAME     = 'vit_small_patch14_dinov2.lvd142m'
DINO_IMAGE_SIZE     = 280
MEAN, STD           = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
HEADERS             = {"User-Agent": "Mozilla/5.0"}
# ══════════════════════════════════════════════════════════════

DIR_YES    = os.path.join(SAVE_DIR, "1_со_средником")
DIR_NO_SR  = os.path.join(SAVE_DIR, "2_отклонено_по_среднику")
DIR_NO_BND = os.path.join(SAVE_DIR, "3_переплёт_не_найден")
LOG_PATH   = os.path.join(SAVE_DIR, "log.csv")
LOG_FIELDS = ['manuscript', 'url', 'n_images', 'binding_scores', 'best_index',
              'best_binding', 'srednik', 'decision', 'time']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pad_transform(size):
    return [A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size,
                          border_mode=0, value=[255, 255, 255])]


# ─────────────── загрузка моделей (тип определяется по весам) ───────────────

def build_dino():
    backbone = timm.create_model(DINO_MODEL_NAME, pretrained=False, num_classes=0,
                                 drop_path_rate=0.1, img_size=DINO_IMAGE_SIZE,
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


def load_all_models(models_dir, device):
    files = sorted(Path(models_dir).glob('*.pth'))
    if not files:
        raise FileNotFoundError(f"В папке '{models_dir}' нет файлов .pth")
    convnext, dino, info = [], [], []
    for p in files:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        sd = ck['model_state_dict']
        if any(k.startswith('backbone.') for k in sd):
            m = build_dino()
            m.load_state_dict(sd)
            dino.append(m.to(device).eval())
            kind, size = 'DINOv2', DINO_IMAGE_SIZE
        elif any(k.startswith('stages.') for k in sd):
            name = ck.get('model_name', 'convnext_tiny.fb_in22k')
            m = timm.create_model(name, pretrained=False, num_classes=2)
            m.load_state_dict(sd)
            size = int(ck.get('image_size', 352))
            convnext.append((m.to(device).eval(), size))
            kind = 'ConvNeXt'
        else:
            print(f"  ⚠ {p.name}: неизвестная модель, пропускаю")
            continue
        info.append({'file': p.name, 'type': kind, 'image_size': size,
                     'fold': ck.get('fold'), 'md5': md5(p)})
        print(f"  {kind:<8} {p.name}  ({size}px)")
    if not convnext or not dino:
        raise RuntimeError(f"Нужны и ConvNeXt, и DINOv2: найдено "
                           f"ConvNeXt={len(convnext)}, DINOv2={len(dino)}")
    return convnext, dino, info


class BindingDetector:
    """Ступень 1: среднее p(переплёт) по всем моделям ConvNeXt, без TTA."""

    def __init__(self, models, device):
        self.models, self.device = models, device
        self.tfm = {size: A.Compose(pad_transform(size) +
                                    [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
                    for _, size in models}

    @torch.no_grad()
    def score(self, img):
        probs = []
        for model, size in self.models:
            x = self.tfm[size](image=img)['image'].unsqueeze(0).to(self.device)
            probs.append(F.softmax(model(x).float(), dim=1)[0, 1].item())
        return float(np.mean(probs))


class SrednikDetector:
    """Ступень 2: среднее p(средник) по всем моделям DINOv2 × 5 TTA."""

    def __init__(self, models, device):
        self.models, self.device = models, device
        sz = DINO_IMAGE_SIZE
        norm = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
        self.tta = [
            A.Compose(pad_transform(sz) + norm),
            A.Compose(pad_transform(sz) + [A.HorizontalFlip(p=1.0)] + norm),
            A.Compose(pad_transform(sz) + [A.VerticalFlip(p=1.0)] + norm),
            A.Compose([A.LongestMaxSize(max_size=int(sz * 1.05)),
                       A.PadIfNeeded(min_height=sz, min_width=sz,
                                     border_mode=0, value=[255, 255, 255]),
                       A.CenterCrop(height=sz, width=sz)] + norm),
            A.Compose(pad_transform(sz) + [A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + norm),
        ]

    @torch.no_grad()
    def score(self, img):
        probs = []
        for model in self.models:
            for t in self.tta:
                x = t(image=img)['image'].unsqueeze(0).to(self.device)
                probs.append(F.softmax(model(x).float(), dim=1)[0, 1].item())
        return float(np.mean(probs))


# ─────────────── получение изображений (как в v2) ───────────────

def make_driver():
    from selenium.webdriver import Edge
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService
    opts = EdgeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.set_capability("ms:loggingPrefs", {"performance": "ALL"})
    service = EdgeService(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "msedgedriver.exe"))
    return Edge(service=service, options=opts)


def _parse_image_logs(logs, seen, urls, n):
    for entry in logs:
        try:
            msg = json.loads(entry["message"])["message"]
            if msg.get("method") == "Network.responseReceived":
                url = msg.get("params", {}).get("response", {}).get("url", "")
                if "img.lib-fond.ru" in url and "PREVIEW" in url:
                    full = url.replace("JPG_PREVIEW", "JPG")
                    if full not in seen:
                        seen.add(full)
                        urls.append(full)
                        if len(urls) >= n:
                            return True
        except Exception:
            pass
    return False


def get_urls_via_selenium(item_url, n):
    driver = make_driver()
    try:
        driver.get(item_url)
        seen, urls = set(), []
        time.sleep(3)
        deadline = time.time() + WAIT_TIME
        scroll_pos = 0
        while time.time() < deadline:
            if _parse_image_logs(driver.get_log("performance"), seen, urls, n):
                break
            scroll_pos += 400
            driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
            time.sleep(1.5)
            if scroll_pos >= driver.execute_script("return document.body.scrollHeight"):
                time.sleep(2)
                _parse_image_logs(driver.get_log("performance"), seen, urls, n)
                break
        return urls
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def guess_urls_direct(first_url, n):
    m = re.search(r'DSC_(\d+)\.JPG', first_url, re.IGNORECASE)
    if not m:
        return None
    start = int(m.group(1))
    base, ext = first_url[:m.start()], first_url[m.end():]
    return [f"{base}DSC_{start + i:04d}.JPG{ext}" for i in range(n)]


def fetch_bytes(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def collect_items():
    slug = COLLECTION_NAME.lower().replace('-', '')
    # Шифр рукописи берётся целиком: f-122-52-3 -> «52-3» (картон 52, ед. хр. 3),
    # f-272-208 -> «208». Раньше бралось только первое число, и все единицы
    # одного картона склеивались в одну.
    pattern = re.compile(r'(?:https?://lib-fond\.ru)?/lib-rgb/' + re.escape(COLLECTION_NAME.lower())
                         + r'/f-' + re.escape(slug) + r'-([0-9][^/?#"]*)/?', re.IGNORECASE)
    items, seen, empty_streak = [], set(), 0
    for page in range(START_PAGE, END_PAGE + 1):
        url = f"{BASE_COLLECTION_URL}?page={page}"
        new, status, n_links, html = 0, None, 0, ''
        for attempt in range(3):  # до 3 попыток на страницу
            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                status, html = r.status_code, r.text
            except Exception as e:
                status, html = f'ошибка {e}', ''
            soup = BeautifulSoup(html, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True)]
            n_links = len(links)
            for href in links:
                m = pattern.search(href)
                if m and m.group(1).lower() not in seen:
                    num = m.group(1).lower()
                    seen.add(num)
                    items.append((num, href if href.startswith('http') else 'https://lib-fond.ru' + href))
                    new += 1
            if new:
                break
            time.sleep(3 * (attempt + 1))
        print(f"  Страница {page}: +{new} рукописей")
        if new == 0:
            empty_streak += 1
            dbg = os.path.join(SAVE_DIR, f"debug_page_{page}.html")
            with open(dbg, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"    пусто (ответ {status}, ссылок на странице {n_links}); HTML сохранён: {dbg}")
            if empty_streak >= 3:
                break
        else:
            empty_streak = 0
        time.sleep(1)
    print(f"  Всего рукописей в фонде: {len(items)}\n")
    return items


# ─────────────── обработка одной рукописи ───────────────

def save(path, data):
    with open(path, 'wb') as f:
        f.write(data)


def process_item(item, binder, srednik):
    num, url = item
    row = {'manuscript': num, 'url': url, 'n_images': 0, 'binding_scores': '',
           'best_index': '', 'best_binding': '', 'srednik': '', 'decision': '',
           'time': datetime.datetime.now().isoformat(timespec='seconds')}
    print(f"  [{num}] {url}")

    urls = get_urls_via_selenium(url, n=IMAGES_TO_CHECK)
    if not urls:
        row['decision'] = 'нет_изображений'
        print("    ✗ изображения не найдены")
        return row
    guessed = guess_urls_direct(urls[0], IMAGES_TO_CHECK)
    if guessed:
        urls = guessed

    scores, best = [], (-1.0, None, -1)
    for i, u in enumerate(urls):
        data = fetch_bytes(u)
        if data is None:
            scores.append('nan')
            continue
        try:
            img = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
        except Exception:
            scores.append('nan')
            continue
        s = binder.score(img)
        scores.append(f"{s:.4f}")
        if s > best[0]:
            best = (s, data, i, img)
        if s >= EARLY_STOP_CONFIDENCE:
            break
        time.sleep(REQUEST_DELAY)

    row['n_images'] = len(scores)
    row['binding_scores'] = ';'.join(scores)
    if best[1] is None:
        row['decision'] = 'нет_изображений'
        print("    ✗ не удалось скачать изображения")
        return row

    b_score, data, idx, img = best
    row['best_index'], row['best_binding'] = idx + 1, f"{b_score:.4f}"
    fname = f"ОР_РГБ_{COLLECTION_NAME}_{num}.jpg"

    if b_score < BINDING_MIN_SCORE:
        row['decision'] = 'переплёт_не_найден'
        save(os.path.join(DIR_NO_BND, f"b{b_score:.2f}__{fname}"), data)
        print(f"    ✗ переплёт не найден (макс. {b_score:.1%})")
        return row

    s_score = srednik.score(img)
    row['srednik'] = f"{s_score:.4f}"
    if s_score >= SREDNIK_MIN_SCORE:
        row['decision'] = 'со_средником'
        save(os.path.join(DIR_YES, fname), data)
        print(f"    ✓ переплёт {b_score:.1%}, средник {s_score:.1%} → сохранено")
    else:
        row['decision'] = 'отклонено_по_среднику'
        save(os.path.join(DIR_NO_SR, f"s{s_score:.2f}__{fname}"), data)
        print(f"    ✗ переплёт {b_score:.1%}, средник {s_score:.1%} → отклонено")
    return row


def main():
    try:
        import selenium  # noqa: F401
    except ImportError:
        raise SystemExit("Не установлен selenium: pip install selenium beautifulsoup4 requests")
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "msedgedriver.exe")):
        raise SystemExit("Рядом со скриптом нет msedgedriver.exe")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for d in (DIR_YES, DIR_NO_SR, DIR_NO_BND):
        os.makedirs(d, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    print(f"\nМодели из {MODELS_DIR}:")
    convnext, dino, info = load_all_models(MODELS_DIR, device)
    binder, srednik = BindingDetector(convnext, device), SrednikDetector(dino, device)
    print(f"  => ConvNeXt: {len(convnext)} | DINOv2: {len(dino)} (× 5 TTA)")

    with open(os.path.join(SAVE_DIR, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'collection': COLLECTION_NAME, 'started': datetime.datetime.now().isoformat(),
                   'tau1': BINDING_MIN_SCORE, 'tau2': SREDNIK_MIN_SCORE,
                   'p_es': EARLY_STOP_CONFIDENCE, 'images_to_check': IMAGES_TO_CHECK,
                   'torch': torch.__version__, 'timm': timm.__version__,
                   'albumentations': A.__version__, 'models': info},
                  f, ensure_ascii=False, indent=2)

    print(f"\nСбор списка рукописей фонда {COLLECTION_NAME}...")
    items = collect_items()
    done = set()
    if os.path.exists(LOG_PATH):
        # строки с ошибками выкидываем, чтобы эти рукописи обработались заново
        with open(LOG_PATH, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        good = [r for r in rows if not r['decision'].startswith('ошибка')]
        if len(good) != len(rows):
            print(f"  Из журнала убрано строк с ошибками: {len(rows) - len(good)}")
            with open(LOG_PATH, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                w.writeheader()
                w.writerows(good)
        done = {r['manuscript'] for r in good}
    todo = [it for it in items if it[0] not in done]
    print(f"Уже обработано: {len(done)} | осталось: {len(todo)}\n")

    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        for k, it in enumerate(todo, 1):
            print(f"({k}/{len(todo)})", end=' ')
            try:
                row = process_item(it, binder, srednik)
            except Exception as e:
                print(f"    ⚠ ошибка: {e}")
                row = {'manuscript': it[0], 'url': it[1], 'decision': f'ошибка: {e}'[:120],
                       'time': datetime.datetime.now().isoformat(timespec='seconds')}
            w.writerow(row)
            f.flush()

    with open(LOG_PATH, encoding='utf-8') as f:
        from collections import Counter
        c = Counter(r['decision'] for r in csv.DictReader(f))
    print(f"\nГотово. Итог по фонду {COLLECTION_NAME}:")
    for k, v in c.most_common():
        print(f"  {k}: {v}")
    print(f"Результаты: {SAVE_DIR}/")


if __name__ == '__main__':
    main()
