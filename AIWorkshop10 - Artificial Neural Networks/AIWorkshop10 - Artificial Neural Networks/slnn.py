import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Load input data
text = np.loadtxt('data_simple_nn.txt')

# Separate it into datapoints and labels (one-hot targets)
data = text[:, 0:2]
labels = text[:, 2:]

# Plot input data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels.argmax(axis=1))
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# Single-layer NN: 2 output neurons (logistic activation)
nn = MLPRegressor(
    hidden_layer_sizes=(2,),
    activation='logistic',
    max_iter=2000,
    learning_rate_init=0.05,
    random_state=42,
)
nn.fit(data, labels)
error_progress = nn.loss_curve_

# Plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()

print('\nTest results:')
data_test = [[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    pred = nn.predict([item])[0]
    pred = np.round(np.clip(pred, 0, 1), 1)
    print(item, '-->', pred)
