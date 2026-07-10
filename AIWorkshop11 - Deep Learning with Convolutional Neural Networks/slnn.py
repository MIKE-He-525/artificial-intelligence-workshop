import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.datasets import mnist 

# Get the MNIST data
batch_size = 90
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_dataset.shuffle(10000).batch(batch_size).repeat()
test_db = test_dataset.batch(batch_size)

# Define model
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.flatten(x)
        y = self.dense(x)
        return y

model = Model()

# Define the gradient descent optimizer
optimizer = tf.keras.optimizers.SGD(0.5)

# Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define the accuracy metric
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()

# Start training
num_iterations = 1200
for i, (x_data, y_data) in enumerate(train_db.take(num_iterations)):
    with tf.GradientTape() as tape:
        # Train the model
        y = model(x_data)
        # Define the entropy loss
        loss = loss_fn(y_data, y)
    # Calculate the gradient
    grads = tape.gradient(loss, model.trainable_variables)
    # Optimizing model parameters
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Update accuracy
    accuracy_fn.update_state(y_data, y)
    print('Iteration', i, ', Accuracy =', accuracy_fn.result().numpy())

# Compute the accuracy using test data
accuracy_fn.reset_state()
for x_data, y_data in test_db:
    y = model(x_data)
    accuracy_fn.update_state(y_data, y)
print('\nAccuracy =', accuracy_fn.result().numpy())

