import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 20 common Chinese characters (Workshop 10.6 Q5)
CHN_CHARS = '一二三四五六七八九十人口大小中国天地山水'

HEIGHT, WIDTH = 16, 8
NUM_CHARS = len(CHN_CHARS)
INPUT_DIM = HEIGHT * WIDTH

FONT_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\simsun.ttc',
    r'C:\Windows\Fonts\msyhbd.ttc',
    'msyh.ttc',
    'simhei.ttf',
    'simsun.ttc',
]


def load_chinese_font(size=64):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError('No Chinese font found. Install Microsoft YaHei / SimHei / SimSun.')


def render_char_bitmap(ch, font, canvas_size=160, out_size=(WIDTH, HEIGHT)):
    """Render bitmap matching letter.data: strokes=1, background=0."""
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size - tw) // 2
    y = (canvas_size - th) // 2
    draw.text((x, y), ch, fill=0, font=font)
    img = img.resize(out_size, Image.Resampling.LANCZOS)
    return 1.0 - np.asarray(img, dtype=np.float32).flatten() / 255.0


def augment_sample(base, noise_std=0.02):
    sample = base + np.random.normal(0, noise_std, base.shape)
    return np.clip(sample, 0.0, 1.0)


def build_dataset(samples_per_char=40, noise_std=0.02):
    font = load_chinese_font()
    data, labels = [], []

    for idx, ch in enumerate(CHN_CHARS):
        base = render_char_bitmap(ch, font)
        if base.max() < 0.1:
            raise RuntimeError(f'Character "{ch}" did not render correctly.')

        for _ in range(samples_per_char):
            data.append(augment_sample(base, noise_std))
            labels.append(idx)

    return np.asarray(data), np.asarray(labels)


def stratified_split(data, labels, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []

    for idx in range(NUM_CHARS):
        char_indices = np.where(labels == idx)[0]
        rng.shuffle(char_indices)
        n_test = max(1, int(len(char_indices) * test_ratio))
        test_idx.extend(char_indices[:n_test].tolist())
        train_idx.extend(char_indices[n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]


if __name__ == '__main__':
    np.random.seed(42)
    data, labels = build_dataset()
    x_train, y_train, x_test, y_test = stratified_split(data, labels)

    # Same idea as ocr_characters.py: feedforward NN with hidden layers.
    # sklearn MLP is used here for stability with modern NumPy (neurolab has
    # compatibility issues and slow convergence on 20-class OCR).
    model = MLPClassifier(
        hidden_layer_sizes=(128, 32),
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
    )

    print(f'Training Chinese OCR on {NUM_CHARS} characters '
          f'({len(x_train)} train / {len(x_test)} test, {INPUT_DIM} features)...')
    print(f'Backend: sklearn.neural_network.MLPClassifier')
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    print('\nTesting on held-out samples:')
    for true_idx, pred_idx in zip(y_test, predicted):
        print('Original:', CHN_CHARS[true_idx], '| Predicted:', CHN_CHARS[pred_idx])

    accuracy = accuracy_score(y_test, predicted)
    print(f'\nTest accuracy: {accuracy:.2%} ({int(accuracy * len(y_test))}/{len(y_test)})')
    print('Note: Chinese OCR is harder than English letters due to larger '
          'character set and stroke complexity.')
