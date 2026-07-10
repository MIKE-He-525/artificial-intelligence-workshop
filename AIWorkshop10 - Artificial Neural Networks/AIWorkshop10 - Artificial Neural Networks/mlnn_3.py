import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Generate some training data
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

plt.figure()
plt.scatter(data, labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# Multilayer NN with 3 hidden layers: 12 -> 8 -> 4 -> 1
nn = MLPRegressor(
    hidden_layer_sizes=(12, 8, 4),
    activation='relu',
    solver='adam',
    max_iter=2000,
    learning_rate_init=0.001,
    random_state=42,
)
nn.fit(data, labels.ravel())
error_progress = nn.loss_curve_

output = nn.predict(data)
y_pred = output.reshape(num_points)

plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')

x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.predict(x_dense.reshape(-1, 1))

plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('Actual vs predicted')

plt.show()
