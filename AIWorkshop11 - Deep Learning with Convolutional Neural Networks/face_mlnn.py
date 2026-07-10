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

# -------------------------- 2. Define MLNN Model --------------------------
class YaleMLNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(YaleMLNN, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(*img_size, 1))
        # Hidden layers with ReLU activation (adds non-linearity)
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")  # 4096 → 512
        self.dense2 = tf.keras.layers.Dense(256, activation="relu")  # 512 → 256
        self.dropout = tf.keras.layers.Dropout(0.3)  # Prevent overfitting
        self.output_layer = tf.keras.layers.Dense(num_classes)  # 256 → 15
    
    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)  # Dropout only active during training
        y_pred = self.output_layer(x)
        return y_pred

model = YaleMLNN(num_classes)

# -------------------------- 3. Training Configuration --------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)  # Lower LR for stable training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

# -------------------------- 4. Train Model --------------------------
num_iterations = 400
train_acc_history = []
test_acc_history = []

print("Training MLNN for Yale Face Recognition...")
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
print(f"\nMLNN Final Test Accuracy: {final_test_acc:.4f}")

# -------------------------- 6. Plot Performance --------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), train_acc_history, label="MLNN Train Accuracy", color="green")
plt.plot(range(20, num_iterations + 1, 20), test_acc_history, label="MLNN Test Accuracy", color="orange", marker="s")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("MLNN Training & Test Accuracy (Yale Faces)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("mlnn_yale_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# Save model
model.save_weights("mlnn_yale_weights.weights.h5")