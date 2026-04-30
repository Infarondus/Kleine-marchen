"""
Двухступенчатый скрапер переплётов со средником

Ступень 1 — ConvNeXt: проверяет что изображение является переплётом
Ступень 2 — DINOv2 (ансамбль 5 фолдов): проверяет наличие средника

Логика для каждой рукописи:
  1. Скачиваем первые IMAGES_TO_CHECK изображений
  2. ConvNeXt отбирает лучший переплёт (score >= BINDING_MIN_SCORE)
  3. DINOv2 проверяет средник на отобранном переплёте
  4. Если score средника >= SREDNIK_MIN_SCORE — сохраняем, иначе пропускаем
"""

import os
import re
import io
import json
import time
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from pathlib import Path
from bs4 import BeautifulSoup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# CONFIG
BASE_COLLECTION_URL = "https://lib-fond.ru/lib-rgb/272"
COLLECTION_NAME     = "272"

START_PAGE  = 1
END_PAGE    = 10

SAVE_DIR    = "images/bindings_srednik"

#Ступень 1
CONVNEXT_MODEL_PATH = "./data/raw/models/binding_model.pth"
CONVNEXT_IMAGE_SIZE = 320
BINDING_MIN_SCORE   = 0.6    # нижняя планка для отбора

#Ступень 2
DINO_MODELS_DIR  = "./kleine-marchen/models"              
DINO_MODEL_NAME  = 'vit_small_patch14_dinov2.lvd142m'
DINO_IMAGE_SIZE  = 280
DINO_DROP_PATH   = 0.1
SREDNIK_MIN_SCORE = 0.55   # ниже нет уверенности в среднике

#Общие настройки
IMAGES_TO_CHECK       = 5
WAIT_TIME             = 20     
EARLY_STOP_CONFIDENCE = 0.95   
REQUEST_DELAY         = 0.3    
WORKERS               = 1

HEADERS = {"User-Agent": "Mozilla/5.0"}

os.makedirs(SAVE_DIR, exist_ok=True)


#ступень 1: ConvNeXt

class BindingDetector:
    """ConvNeXt: определяет является ли изображение переплётом."""

    def __init__(self, model_path):
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.available = os.path.exists(model_path)

        if self.available:
            checkpoint  = torch.load(model_path, map_location=self.device,
                                     weights_only=False)
            image_size  = checkpoint.get('image_size', CONVNEXT_IMAGE_SIZE)
            self.model  = timm.create_model('convnext_tiny_in22k',
                                            pretrained=False, num_classes=2)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device).eval()
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=image_size),
                A.PadIfNeeded(min_height=image_size, min_width=image_size,
                              border_mode=0, value=[255, 255, 255]),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
            print(f"  ConvNeXt загружен ({self.device}): {model_path}")
        else:
            print(f"  ⚠ ConvNeXt не найден: {model_path} — ступень 1 пропускается")

    def score(self, img_bytes) -> float | None:
        if not self.available:
            return None
        try:
            img    = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            tensor = self.transform(image=np.array(img))['image'] \
                         .unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(tensor), dim=1)[0]
            return float(probs[1].item())
        except Exception as e:
            print(f"      ConvNeXt ошибка: {e}")
            return None



#ступень 2: ансамбль моделей DINOv2

def _build_dino_model(device):
    backbone = timm.create_model(
        DINO_MODEL_NAME, pretrained=False, num_classes=0,
        drop_path_rate=DINO_DROP_PATH,
        img_size=DINO_IMAGE_SIZE, dynamic_img_size=True,
    )
    head = nn.Sequential(
        nn.LayerNorm(backbone.embed_dim),
        nn.Linear(backbone.embed_dim, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )

    class DinoClassifier(nn.Module):
        def __init__(self, b, h):
            super().__init__()
            self.backbone = b
            self.head = h
        def forward(self, x):
            return self.head(self.backbone(x))

    return DinoClassifier(backbone, head).to(device)


class SrednikDetector:
    DINO_MEAN = [0.485, 0.456, 0.406]
    DINO_STD  = [0.229, 0.224, 0.225]

    def __init__(self, models_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_files = sorted(Path(models_dir).glob('fold_*_dinov2.pth'))

        if not model_files:
            raise FileNotFoundError(
                f"DINOv2 модели не найдены в '{models_dir}'.\n"
                f"Убедитесь что папка models/ с fold_*_dinov2.pth рядом со скриптом.")

        self.models = []
        for path in model_files:
            ckpt  = torch.load(path, map_location=self.device, weights_only=False)
            model = _build_dino_model(self.device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            self.models.append(model)

        print(f"  DINOv2 загружен ({self.device}): {len(self.models)} фолдов из {models_dir}")

        sz = DINO_IMAGE_SIZE
        base = [
            A.LongestMaxSize(max_size=sz),
            A.PadIfNeeded(min_height=sz, min_width=sz,
                          border_mode=0, value=[255, 255, 255]),
        ]
        norm = [A.Normalize(mean=self.DINO_MEAN, std=self.DINO_STD), ToTensorV2()]
        self.tta = [
            A.Compose(base + norm),
            A.Compose(base + [A.HorizontalFlip(p=1.0)] + norm),
            A.Compose(base + [A.VerticalFlip(p=1.0)] + norm),
            A.Compose([A.LongestMaxSize(max_size=int(sz * 1.05)),
                       A.PadIfNeeded(min_height=sz, min_width=sz,
                                     border_mode=0, value=[255, 255, 255]),
                       A.CenterCrop(height=sz, width=sz)] + norm),
            A.Compose(base + [A.RandomBrightnessContrast(
                              brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + norm),
        ]

    @torch.no_grad()
    def score(self, img_bytes) -> float | None:
        try:
            img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
        except Exception as e:
            print(f"      DINOv2 ошибка чтения: {e}")
            return None

        all_probs = []
        for model in self.models:
            for tfm in self.tta:
                tensor = tfm(image=img)['image'].unsqueeze(0).to(self.device)
                probs  = F.softmax(model(tensor), dim=1).squeeze(0).cpu()
                all_probs.append(probs)

        avg = torch.stack(all_probs).mean(0)
        return float(avg[1].item())


#selenium — получение URL изображений

def make_driver():
    from selenium.webdriver import Edge
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService

    opts = EdgeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.set_capability("ms:loggingPrefs", {"performance": "ALL"})

    service = EdgeService(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "msedgedriver.exe"))
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
            logs = driver.get_log("performance")
            if _parse_image_logs(logs, seen, urls, n):
                break
            if len(urls) >= n:
                break
            scroll_pos += 400
            driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
            time.sleep(1.5)
            page_height = driver.execute_script("return document.body.scrollHeight")
            if scroll_pos >= page_height:
                time.sleep(2)
                _parse_image_logs(driver.get_log("performance"), seen, urls, n)
                break
        return urls
    finally:
        try: driver.quit()
        except Exception: pass


def guess_urls_direct(first_url, n):
    m = re.search(r'DSC_(\d+)\.JPG', first_url, re.IGNORECASE)
    if not m:
        return None
    start = int(m.group(1))
    base, ext = first_url[:m.start()], first_url[m.end():]
    return [f"{base}DSC_{start+i:04d}.JPG{ext}" for i in range(n)]


def fetch_bytes(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


#парсинг списка рукописей

def collect_items():
    slug    = COLLECTION_NAME.lower().replace('-', '')
    pattern = re.compile(
        r'https?://lib-fond\.ru/lib-rgb/' +
        re.escape(COLLECTION_NAME.lower()) +
        r'/f-' + re.escape(slug) + r'-(\d+)/?',
        re.IGNORECASE,
    )
    items = []
    for page in range(START_PAGE, END_PAGE + 1):
        url = BASE_COLLECTION_URL + f"?page={page}"
        print(f"  Страница {page}: {url}")
        try:
            r    = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            seen = set()
            for a in soup.find_all('a', href=True):
                m = pattern.search(a['href'])
                if m:
                    num = int(m.group(1))
                    if num not in seen:
                        seen.add(num)
                        href = a['href']
                        full = href if href.startswith('http') \
                               else 'https://lib-fond.ru' + href
                        items.append((num, full))
        except Exception as e:
            print(f"    Ошибка: {e}")
    print(f"  Найдено рукописей: {len(items)}\n")
    return items


# обработка одной рукописи

binding_detector = None
srednik_detector = None


def process_item(item_data):
    item_num, item_url = item_data

    out_filename = f"ОР_РГБ_{COLLECTION_NAME}_{item_num}.jpg"
    out_path     = os.path.join(SAVE_DIR, out_filename)
    if os.path.exists(out_path):
        print(f"  [{item_num}] пропущено (уже есть)")
        return

    print(f"  [{item_num}] {item_url}")

    #получение url изображений
    all_urls = get_urls_via_selenium(item_url, n=IMAGES_TO_CHECK)
    if not all_urls:
        print(f"    ✗ Selenium не нашёл изображений")
        return

    print(f"    Первый URL: {unquote(all_urls[0])[-60:]}")

    guessed = guess_urls_direct(all_urls[0], IMAGES_TO_CHECK)
    if guessed:
        print(f"    Режим: прямой перебор DSC-номеров")
        all_urls = guessed
    else:
        print(f"    Режим: Selenium ({len(all_urls)} URL)")

    #ищем лучший переплёт
    best_bytes = None
    best_binding_score = -1.0
    best_index = -1

    for i, url in enumerate(all_urls):
        img_bytes = fetch_bytes(url)
        if img_bytes is None:
            print(f"    Изображение {i+1}: недоступно")
            continue

        b_score = binding_detector.score(img_bytes)

        if b_score is None:
            # ConvNeXt недоступен — берём первое изображение
            best_bytes, best_binding_score, best_index = img_bytes, 1.0, i
            break

        print(f"    Изображение {i+1}: переплёт={b_score:.1%}", end="")

        if b_score > best_binding_score:
            best_binding_score = b_score
            best_bytes         = img_bytes
            best_index         = i

        if b_score >= EARLY_STOP_CONFIDENCE:
            print(f" — уверен, останавливаемся")
            break
        else:
            print()

        time.sleep(REQUEST_DELAY)

    if best_bytes is None:
        print(f"    ✗ Не удалось скачать ни одного изображения")
        return

    if best_binding_score < BINDING_MIN_SCORE:
        print(f"    ✗ Переплёт не найден "
              f"(макс. {best_binding_score:.1%} < {BINDING_MIN_SCORE:.0%})")
        return

    print(f"    ✓ Переплёт: изображение №{best_index+1}, "
          f"уверенность {best_binding_score:.1%}")

    #Ступень 2:
    s_score = srednik_detector.score(best_bytes)

    if s_score is None:
        print(f"    ✗ DINOv2: ошибка при проверке средника")
        return

    print(f"    {'✓' if s_score >= SREDNIK_MIN_SCORE else '✗'} "
          f"Средник: {s_score:.1%} "
          f"({'сохраняем' if s_score >= SREDNIK_MIN_SCORE else 'пропускаем'})")

    if s_score < SREDNIK_MIN_SCORE:
        return

    #Сохраняем
    with open(out_path, 'wb') as f:
        f.write(best_bytes)
    print(f"    → Сохранено: {out_filename}")



def main():
    global binding_detector, srednik_detector

    print("Загружаем модели...")
    binding_detector = BindingDetector(CONVNEXT_MODEL_PATH)
    srednik_detector = SrednikDetector(DINO_MODELS_DIR)

    print(f"\nСбор списка рукописей: {BASE_COLLECTION_URL} "
          f"(страницы {START_PAGE}–{END_PAGE})...")
    items = collect_items()

    if not items:
        print("Рукописи не найдены. Проверьте BASE_COLLECTION_URL.")
        return

    already = sum(
        1 for num, _ in items
        if os.path.exists(os.path.join(SAVE_DIR, f"ОР_РГБ_{COLLECTION_NAME}_{num}.jpg"))
    )
    todo = len(items) - already
    print(f"Всего: {len(items)} | уже скачано: {already} | осталось: {todo}")
    print(f"Порог переплёта: {BINDING_MIN_SCORE:.0%} | "
          f"Порог средника: {SREDNIK_MIN_SCORE:.0%}\n")

    if WORKERS > 1:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            ex.map(process_item, items)
    else:
        for item in items:
            process_item(item)

    saved = sum(
        1 for num, _ in items
        if os.path.exists(os.path.join(SAVE_DIR, f"ОР_РГБ_{COLLECTION_NAME}_{num}.jpg"))
    )
    print(f"\nГотово. Сохранено переплётов со средником: {saved}")
    print(f"Папка: {SAVE_DIR}/")


if __name__ == '__main__':
    main()
