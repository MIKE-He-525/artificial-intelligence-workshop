# The following code is using Tensorflow 2.17.0, if you are using Tensorflow 1.X please refer to the AIWorkshop11_old.ipynb file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Define the number of points to generate
num_points = 1200

# Generate the data based on equation y = mx + c
data = []
m = 0.2
c = 0.5
for i in range(num_points):
    x = np.random.normal(0.0, 1.0)  
    noise = np.random.normal(0.0, 0.05)  
    
    y = 1 / (1 + np.exp(-x)) + noise
    data.append([x, y])

# Separate x and y
x_data = [d[0] for d in data]
y_data = [d[1] for d in data]

# Convert to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Plot the generated data
plt.plot(x_data, y_data, 'ro')
plt.title('Input data')
plt.show()

# Define keras model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),  
    tf.keras.layers.Dense(1)  
])


# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Start iterating
num_iterations = 10
for step in range(num_iterations):
    # Train the model
    model.fit(x_data.reshape(-1, 1), y_data, verbose=0)

    # Print the progress
    print('\nITERATION', step+1)
    print('W =', model.layers[0].weights[0].numpy()[0][0])
    print('b =', model.layers[0].weights[1].numpy()[0])
    print('loss =', model.loss)

    # Plot the input data 
    plt.plot(x_data, y_data, 'ro')

    # Plot the predicted output line
    plt.plot(x_data, model.predict(x_data.reshape(-1, 1)).flatten())

    # Set plotting parameters
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.title('Iteration ' + str(step+1) + ' of ' + str(num_iterations))
    plt.show()