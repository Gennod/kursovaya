# Импорты
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from tpot import TPOTClassifier
import autokeras as ak

# Параметры
IMG_SIZE = 128
CATEGORIES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# === Функции ===

def load_and_preprocess_data_from_folder(data_dir, categories, img_size):
    """
    Загружает и обрабатывает изображения из указанной папки.
    """
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_label = categories.index(category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_size, img_size))
                data.append([image, class_label])
            except Exception as e:
                print(f"Ошибка при загрузке {img_name}: {e}")
    return data


def load_and_preprocess_data_from_npz(npz_path):
    """
    Загружает данные из файла .npz.
    """
    data = np.load(npz_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    return X_train, X_test, y_train, y_test


def prepare_data(data, categories):
    """
    Подготавливает данные для модели.
    """
    np.random.shuffle(data)
    X = np.array([item[0] for item in data])
    y = np.array([item[1] for item in data])
    X = X / 255.0
    y = to_categorical(y, num_classes=len(categories))
    return train_test_split(X, y, test_size=0.2, random_state=42)


# === Основной код ===

def main(data_dir=None, npz_path=None):
    """
    Запускает тестирование моделей TPOT и AutoKeras.
    """
    if data_dir:
        print("Загрузка данных из папки...")
        data = load_and_preprocess_data_from_folder(data_dir, CATEGORIES, IMG_SIZE)
        X_train, X_test, y_train, y_test = prepare_data(data, CATEGORIES)
    elif npz_path:
        print("Загрузка данных из .npz файла...")
        X_train, X_test, y_train, y_test = load_and_preprocess_data_from_npz(npz_path)
    else:
        raise ValueError("Необходимо указать data_dir или npz_path.")

    print("Данные загружены и обработаны.")

    # === AutoKeras ===
    print("Запуск AutoKeras...")
    clf = ak.ImageClassifier(overwrite=True, max_trials=2)
    clf.fit(X_train, y_train, epochs=10)
    y_pred_automl = clf.predict(X_test)
    accuracy_automl = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_automl, axis=1))
    print(f"AutoKeras Test Accuracy: {accuracy_automl:.4f}")

    # === TPOT ===
    print("Запуск TPOT...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_train_flat = np.argmax(y_train, axis=1)
    y_test_flat = np.argmax(y_test, axis=1)

    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        verbosity=2,
        max_time_mins=60,
        random_state=42,
        config_dict="TPOT light",
        n_jobs=-1
    )
    tpot.fit(X_train_flat, y_train_flat)
    y_pred_tpot = tpot.predict(X_test_flat)
    accuracy_tpot = accuracy_score(y_test_flat, y_pred_tpot)
    print(f"TPOT Test Accuracy: {accuracy_tpot:.4f}")

    tpot.export("models/tpot_best_pipeline.py")

    # === Сравнение ===
    print(f"\nСравнение результатов:")
    print(f"AutoKeras Accuracy: {accuracy_automl:.4f}")
    print(f"TPOT Accuracy: {accuracy_tpot:.4f}")


# === Запуск ===
if __name__ == "__main__":
    # Укажите путь к данным (папка с изображениями или файл .npz)
    DATA_DIR = None  # Укажите папку с изображениями, если требуется
    NPZ_PATH = "processed_data.npz"  # Укажите путь к .npz файлу, если требуется

    main(data_dir=DATA_DIR, npz_path=NPZ_PATH)
