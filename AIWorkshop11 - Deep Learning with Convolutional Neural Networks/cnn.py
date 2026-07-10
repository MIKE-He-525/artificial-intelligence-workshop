import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Input

# Get the MNIST data
batch_size = 75
(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_db = train_dataset.shuffle(10000).batch(batch_size).repeat()
test_db = test_dataset.batch(batch_size)

# Define model
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.model = tf.keras.Sequential([
            Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'),
            tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])

    def call(self, inputs):
       return self.model(inputs)

model = Model()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Define the accuracy metric
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()

# Start training
num_iterations = 100
print('\nTraining the model....')
for i in range(num_iterations):
    accuracy_fn.reset_state()
    # Get the next batch of images
    for j, (x_data, y_data) in enumerate(train_db.take(1)):
        with tf.GradientTape() as tape:
            # Train the model
            y = model(x_data, training=True)
            # Define the entropy loss
            loss = loss_fn(y_data, y)
        # Calculate the gradient
        grads = tape.gradient(loss, model.trainable_variables)
        # Optimizing model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Update accuracy
        accuracy_fn.update_state(y_data, y)
    print('Iteration', i, ', Accuracy =', accuracy_fn.result().numpy())

# Compute accuracy using test data
accuracy_fn.reset_state()
for x_data, y_data in test_db:
    y = model(x_data)
    accuracy_fn.update_state(y_data, y)
print('Test accuracy =', accuracy_fn.result().numpy())