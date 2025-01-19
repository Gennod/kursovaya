import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from lightautoml.automl.presets.tabular_presets import TabularAutoML
import pickle
import cv2
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

def load_data(data_path):
    """Load data from .npz file."""
    data = np.load(data_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    return X_train, X_test, y_train, y_test

def extract_features(data, model):
    """Extract features using a pre-trained model."""
    features_list = []
    for img in data:
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img).flatten()
        features_list.append(features)
    return np.array(features_list)

def evaluate_model(model, X_test, y_test, model_name, input_format="image"):
    """Evaluate the model and print classification metrics."""
    print(f"\nEvaluating model: {model_name}")

    if input_format == "flattened":
        X_test = X_test.reshape((X_test.shape[0], -1))

    y_pred = model.predict(X_test)
    if y_pred.shape[1] > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()

    print("Classification Report:\n", classification_report(y_test, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

def test_model(model_path, X_test, y_test, model_name, input_format="image"):
    """Test a model if it exists."""
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}. Skipping.")
        return

    model = tf.keras.models.load_model(model_path)
    evaluate_model(model, X_test, y_test, model_name, input_format)

def test_lightautoml_model(model_path, X_test_features, y_test):
    """Test LightAutoML model if it exists."""
    if not os.path.exists(model_path):
        print("LightAutoML model not found. Skipping.")
        return

    print("\nEvaluating model: LightAutoML")
    with open(model_path, 'rb') as f:
        lightautoml_model = pickle.load(f)

    X_test_df = pd.DataFrame(X_test_features, columns=[f'feature_{i}' for i in range(X_test_features.shape[1])])
    predictions = lightautoml_model.predict(X_test_df).data

    if predictions.shape[1] > 1:
        y_pred_classes = np.argmax(predictions, axis=1)
    else:
        y_pred_classes = (predictions > 0.5).astype(int).flatten()

    print("Classification Report:\n", classification_report(y_test, y_pred_classes))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_classes))

def main():
    """Main function to run the evaluations."""
    # Paths to models
    custom_model_path = '/models/corn_disease_classifier.keras'
    autokeras_model_path = '/models/saved_autokeras_model.keras'
    lightautoml_model_path = '/models/saved_automl_model.pkl'
    keras_tuner_model_path = '/models/corn_disease_classifier_NAS(Keras-Tuner).keras'

    # Load data
    X_train_small, X_test_small, y_train_small, y_test_small = load_data("/data/processed_data(small).npz")
    X_train_medium, X_test_medium, y_train_medium, y_test_medium = load_data("/data/processed_data(medium).npz")

    # Load pre-trained ResNet50 for feature extraction
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    # Test models
    test_model(custom_model_path, X_test_small, y_test_small, "Custom Model (Keras)")
    test_model(autokeras_model_path, X_test_small, y_test_small, "AutoKeras Model")

    if os.path.exists(keras_tuner_model_path):
        keras_tuner_model = tf.keras.models.load_model(keras_tuner_model_path)
        X_test_resized = np.array([tf.image.resize(img, (128, 128)) for img in X_test_small])
        evaluate_model(keras_tuner_model, X_test_resized, y_test_small, "Custom Model (Keras-Tuner)")
    else:
        print("Model Custom Model (Keras-Tuner) not found. Skipping.")

    # Extract features and test LightAutoML
    X_test_small_features = extract_features(X_test_small, base_model)
    test_lightautoml_model(lightautoml_model_path, X_test_small_features, y_test_small)

if __name__ == "__main__":
    main()
