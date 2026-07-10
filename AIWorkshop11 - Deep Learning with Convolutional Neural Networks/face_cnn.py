import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_yale_data

# -------------------------- 1. Load Data --------------------------
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Yale", "yalefaces")
img_size = (64, 64)
(X_train, y_train), (X_test, y_test) = load_yale_data(data_dir, img_size)
num_classes = len(np.unique(np.concatenate([y_train, y_test])))

batch_size = 16
train_db = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(batch_size).repeat()
test_db = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# -------------------------- 2. Define CNN Model --------------------------
class YaleCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(YaleCNN, self).__init__()
        self.cnn_layers = tf.keras.Sequential([
            # 1st Convolutional Block: Conv → ReLU → MaxPool
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(*img_size, 1)),
            tf.keras.layers.MaxPool2D((2, 2), strides=2),  # 64x64 → 32x32
            # 2nd Convolutional Block
            tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2), strides=2),  # 32x32 → 16x16
            # 3rd Convolutional Block
            tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D((2, 2), strides=2),  # 16x16 → 8x8
        ])
        # Classification Head
        self.flatten = tf.keras.layers.Flatten()  # 8x8x128 → 8192
        self.dense = tf.keras.layers.Dense(256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(0.4)  # Reduce overfitting
        self.output_layer = tf.keras.layers.Dense(num_classes)
    
    def call(self, x, training=False):
        x = self.cnn_layers(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        y_pred = self.output_layer(x)
        return y_pred

model = YaleCNN(num_classes)

# -------------------------- 3. Training Configuration --------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

# -------------------------- 4. Train Model --------------------------
num_iterations = 500
train_acc_history = []
test_acc_history = []

print("Training CNN for Yale Face Recognition...")
for i in range(num_iterations):
    x_batch, y_batch = next(iter(train_db))
    with tf.GradientTape() as tape:
        y_pred = model(x_batch, training=True)
        loss = loss_fn(y_batch, y_pred)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Track training accuracy
    accuracy_fn.update_state(y_batch, y_pred)
    train_acc = accuracy_fn.result().numpy()
    train_acc_history.append(train_acc)
    
    # Evaluate on test set every 20 iterations
    if (i + 1) % 20 == 0:
        accuracy_fn.reset_state()
        for x_test_batch, y_test_batch in test_db:
            y_test_pred = model(x_test_batch, training=False)
            accuracy_fn.update_state(y_test_batch, y_test_pred)
        test_acc = accuracy_fn.result().numpy()
        test_acc_history.append(test_acc)
        print(f"Iteration {i+1:3d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        accuracy_fn.reset_state()

# -------------------------- 5. Final Test Evaluation --------------------------
accuracy_fn.reset_state()
for x_test_batch, y_test_batch in test_db:
    y_test_pred = model(x_test_batch, training=False)
    accuracy_fn.update_state(y_test_batch, y_test_pred)
final_test_acc = accuracy_fn.result().numpy()
print(f"\nCNN Final Test Accuracy: {final_test_acc:.4f}")

# -------------------------- 6. Plot Performance --------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), train_acc_history, label="CNN Train Accuracy", color="purple")
plt.plot(range(20, num_iterations + 1, 20), test_acc_history, label="CNN Test Accuracy", color="cyan", marker="^")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("CNN Training & Test Accuracy (Yale Faces)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("cnn_yale_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# Save model
model.save_weights("cnn_yale_weights.weights.h5")