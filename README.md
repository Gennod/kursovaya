# Распознавание болезней культурных растений с помощью AutoML

## Описание проекта

Данный проект посвящен автоматизации распознавания болезней культурных растений с помощью AutoML-фреймворков. Целью исследования является сравнение производительности моделей, созданных с помощью AutoKeras и TPOT, на задаче классификации изображений растений по состоянию их листьев.

Проект включает:
* Предобработку данных из изображений или формата .npz.
* Обучение моделей с использованием AutoML-фреймворков.
* Сравнение точности предсказаний моделей TPOT и AutoKeras.


## Используемые инструменты

* Python 3.8+
* AutoKeras
* TPOT
* TensorFlow
* Scikit-learn
* OpenCV

## Данные 

Данные представляют собой изображения листьев растений, разделенные на 4 категории:
* Blight
* Common Rust
* Gray Leaf Spot
* Healthy

Данные могут быть предоставлены в двух форматах:
* Папка с изображениями, разделенными по категориям.
* Архивированный файл .npz с предварительно обработанными массивами.

Данные, которые использовались для проекта: [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset).

## Установка 

1. Клонируйте репозиторий: 

```
git clone https://github.com/Gennod/kursovaya.git cd ваш-репозиторий
```

2. 

```
pip install -r requirements.txt
```

## Запуск

Обработанные данные:

* small (64x64) https://drive.google.com/file/d/1gAc3R9iSRGME9scB2aCj3FkWK52fjYTt/view?usp=sharing
* medium (128x128) https://drive.google.com/file/d/1VrMftJvF3VuUt-k3yNPCChR9qi0GcKKe/view?usp=sharing
* big (255*255) https://drive.google.com/file/d/1G2iGRaYnffXltUB-Fh-L5Z5sL2we3zeZ/view?usp=sharing

### Обработка данных

Для предобработки данных:

* Если у вас папка с изображениями, измените параметр DATA_DIR в файле test_models.py на путь к папке с изображениями.
* Если у вас файл .npz, укажите путь в параметре NPZ_PATH.

### Тестирование модели

Запустите файл test_models.py:

```
python test_models.py
```

Скрипт выполнит следующие шаги:
* Загрузит и обработает данные.
* Обучит модель AutoKeras.
* Обучит модель TPOT.
* Выведет результаты сравнения точности моделей

## Результаты

Результаты работы TPOT:
* Лучший пайплайн будет сохранен в models/tpot_best_pipeline.py

Результаты работы AutoKeras:
* Лучшая модель будет сохранена в models/corn_disease_classifier.keras

Пример логов работы скрипта:

```
AutoKeras Test Accuracy: 0.9123
TPOT Test Accuracy: 0.8956
Comparison - AutoKeras Accuracy: 0.9123, TPOT Accuracy: 0.8956
```

## Структура проекта

```
├── data/                           # Папка с данными
├── models/                         # Сохраненные модели и пайплайны
│   ├── corn_disease_classifier.keras
│   ├── tpot_best_pipeline.py
├── code/
│   ├── requirements.txt            # Зависимости проекта
│   ├── test_models.py              # Основной скрипт для тестирования
│   ├── notebook.ipynb              # Google Colab ноутбук
└── README.md                       # Описание проекта
```

## Автор

* Студент группы 23ПМИ(м)ГОГИИ Бледнов Иван Андреевич
