import os
import re
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

SUBJECT_PATTERN = re.compile(r'^subject(\d+)', re.I)


def _default_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "Yale", "yalefaces")


def _read_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im.convert('L'))
    except Exception:
        return None


def _load_flat_yale_faces(data_dir, img_size):
    X, y = [], []
    for fname in sorted(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        match = SUBJECT_PATTERN.match(fname)
        if not match:
            continue
        subject_idx = int(match.group(1)) - 1
        img = _read_gray_image(fpath)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        y.append(subject_idx)
    return X, y


def _load_folder_yale_faces(data_dir, img_size):
    X, y = [], []
    for subject_idx, subject_folder in enumerate(sorted(os.listdir(data_dir))):
        subject_path = os.path.join(data_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        for img_file in os.listdir(subject_path):
            img_path = os.path.join(subject_path, img_file)
            if not os.path.isfile(img_path):
                continue
            img = _read_gray_image(img_path)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)
            X.append(img)
            y.append(subject_idx)
    return X, y


def _load_olivetti_fallback(img_size):
    from sklearn.datasets import fetch_olivetti_faces
    print("Yale faces not found locally. Using Olivetti faces as a substitute dataset.")
    faces = fetch_olivetti_faces()
    X, y = [], []
    for img, label in zip(faces.images, faces.target):
        img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=-1)
        X.append(img)
        y.append(label)
    return X, y


def load_yale_data(data_dir=None, img_size=(64, 64)):
    """
    Load Yale Facial Database and preprocess images.
    Falls back to Olivetti faces if Yale/yalefaces is empty.
    """
    data_dir = data_dir or _default_data_dir()
    data_dir = os.path.abspath(data_dir)

    if os.path.isdir(data_dir):
        flat_files = [
            f for f in os.listdir(data_dir)
            if os.path.isfile(os.path.join(data_dir, f)) and SUBJECT_PATTERN.match(f)
        ]
        if flat_files:
            X, y = _load_flat_yale_faces(data_dir, img_size)
        else:
            X, y = _load_folder_yale_faces(data_dir, img_size)
    else:
        X, y = [], []

    if not X:
        X, y = _load_olivetti_fallback(img_size)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_yale_data()
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    print(f"Classes in train set: {len(np.unique(y_train))}")
