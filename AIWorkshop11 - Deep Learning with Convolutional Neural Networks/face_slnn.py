import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import load_yale_data

# -------------------------- 1. Load Data --------------------------
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Yale", "yalefaces")
img_size = (64, 64)
(X_train, y_train), (X_test, y_test) = load_yale_data(data_dir, img_size)
num_classes = len(np.unique(np.concatenate([y_train, y_test])))

# Create TensorFlow Dataset for batching
batch_size = 16
train_db = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(batch_size).repeat()
test_db = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# -------------------------- 2. Define SLNN Model --------------------------
class YaleSLNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(YaleSLNN, self).__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(*img_size, 1))
        self.dense = tf.keras.layers.Dense(num_classes)
    
    def call(self, x):
        x = self.flatten(x)
        y_pred = self.dense(x)
        return y_pred

model = YaleSLNN(num_classes)

# -------------------------- 3. Training Configuration --------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

# -------------------------- 4. Train Model --------------------------
num_iterations = 300  # Adjust based on convergence
train_acc_history = []
test_acc_history = []

print("Training SLNN for Yale Face Recognition...")
for i in range(num_iterations):
    # Train on one batch
    x_batch, y_batch = next(iter(train_db))
    with tf.GradientTape() as tape:
        y_pred = model(x_batch, training=True)
        loss = loss_fn(y_batch, y_pred)
    
    # Backpropagation
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
print(f"\nSLNN Final Test Accuracy: {final_test_acc:.4f}")

# -------------------------- 6. Plot Performance --------------------------
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), train_acc_history, label="SLNN Train Accuracy", color="blue")
plt.plot(range(20, num_iterations + 1, 20), test_acc_history, label="SLNN Test Accuracy", color="red", marker="o")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("SLNN Training & Test Accuracy (Yale Faces)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("slnn_yale_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# Save model
model.save_weights("slnn_yale_weights.weights.h5")