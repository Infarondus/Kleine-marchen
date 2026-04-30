### Рукописный переплёт со средником — двухэтапное детектирование
```markdown
Проект решает задачу выявления старинных переплётных крышек, на которых присутствует **средник** (центральный декоративный элемент).

Работает в два этапа:

1. **ConvNeXt Tiny** — определяет, является ли изображение переплётом (binding / not binding).
2. **DINOv2 ViT-Small** (ансамбль из 5 фолдов) — на найденных переплётах проверяет наличие средника (со средником / без средника).

На выходе можно либо разметить уже скачанные изображения, либо в реальном времени скачивать и фильтровать переплёты со средником с сайта рукописного фонда.
```

## Структура репозитория

```bash
.
├── deep_test_v4_dino.py          # обучение DINOv2 (средник/без средника)
├── deep_test_v2.py               # обучение ConvNeXt (переплёт/не переплёт) — устаревший
├── evaluate.py                   # оценка ансамбля DINOv2
├── inference.py                  # разметка неразмеченных изображений
├── scrape_bindings_v2.py         # скачивание + двухэтапная фильтрация на лету
│
├── models/                       # обученные модели DINOv2 (fold_*_dinov2.pth)
├── data/raw/models/              # сюда нужно положить binding_model.pth
│
├── selected/                     # (для обучения) переплёты со средником
├── rejected/                     # (для обучения) переплёты без средника
├── binding/                      # (для ConvNeXt) изображения переплётов
├── not_binding/                  # (для ConvNeXt) изображения НЕ переплётов
│
├── eval_results/                 # результаты работы evaluate.py
├── result/                       # результаты работы inference.py
└── images/bindings_srednik/      # сохранённые переплёты со средником (scrape)
```

> **Важно:** Папки `selected`, `rejected`, `binding` и `not_binding` нужно создать самостоятельно и наполнить данными для обучения.

---

## Зависимости

- Python ≥ 3.10
- PyTorch ≥ 2.0
- torchvision
- timm
- albumentations
- scikit-learn
- numpy
- pillow
- beautifulsoup4
- selenium (только для `scrape_bindings_v2.py`)
- requests

### Установка

```bash
pip install torch torchvision timm albumentations scikit-learn numpy pillow beautifulsoup4 selenium requests
```

> Для Windows в `scrape_bindings_v2.py` используется Edge WebDriver. При необходимости замените на ChromeDriver.

---

## Обучение моделей

### 1. ConvNeXt Tiny — детектор переплёта

**Скрипт:** `deep_test_v2.py`  
**Вход:** папки `./binding` и `./not_binding`

**Особенность:** Лучшее качество достигается при обучении **только классифицирующей головы** (фаза 1). Полная разморозка backbone ухудшает результат.

**Запуск:**

```bash
python deep_test_v2.py
```

После обучения скопируйте лучшую модель в `./data/raw/models/binding_model.pth`.

---

### 2. DINOv2 ViT-Small — детектор средника

**Скрипт:** `deep_test_v4_dino.py`  
**Вход:** папки `./selected` и `./rejected`

**Модель:** `vit_small_patch14_dinov2.lvd142m` (21M параметров)  
**Стратегия:** двухфазное обучение + EMA + TTA + взвешенный семплер + label smoothing.

**Запуск:**

```bash
python deep_test_v4_dino.py
```

В результате в папке `models/` появятся 5 файлов: `fold_1_dinov2.pth` … `fold_5_dinov2.pth`.

---

## Оценка качества DINOv2

**Скрипт:** `evaluate.py`

```bash
python evaluate.py
```

Скрипт прогоняет ансамбль с TTA, выводит метрики и сохраняет ошибки + сомнительные изображения в `./eval_results/`.

---

## Инференс (разметка изображений)

**Скрипт:** `inference.py`

```bash
python inference.py --input путь/к/папке --threshold 0.8
```

Изображения будут разложены в `./result/` на три папки:
- `со_средником/`
- `без_средника/`
- `сомнительные/`

---

## Скрапинг с фильтрацией в реальном времени

**Скрипт:** `scrape_bindings_v2.py`

Скрипт скачивает рукописи с `lib-fond.ru`, последовательно применяет обе модели и сохраняет только переплёты со средником.

**Требования:**
- Рядом должен лежать `msedgedriver.exe` (или заменить драйвер)
- Модель ConvNeXt: `./data/raw/models/binding_model.pth`
- Модели DINOv2: `./models/fold_*_dinov2.pth`

```bash
python scrape_bindings_v2.py
```

Настройки (пороги, коллекция и т.д.) задаются в начале файла.

---

## Важные замечания

- ConvNeXt-модель **не рекомендуется** полноценно fine-tune'ить — разморозка слоёв сильно падает качество.
- Для обучения DINOv2 желателен GPU с ≥ 6 ГБ VRAM (batch size 32 при разрешении 280px).
- Все скрипты с DINOv2 ожидают модели в папке `models/` с именами `fold_*_dinov2.pth`.
- При проблемах с Selenium можно переписать парсинг изображений.
