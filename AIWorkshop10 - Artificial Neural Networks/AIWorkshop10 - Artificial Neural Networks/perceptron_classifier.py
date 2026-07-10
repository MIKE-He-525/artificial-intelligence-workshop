import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Load input data
text = np.loadtxt('data_perceptron.txt')

# Separate datapoints and labels
data = text[:, :2]
labels = text[:, 2].astype(int)

# Plot input data
plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# Perceptron classifier (sklearn replacement for neurolab.net.newp)
perceptron = Perceptron(max_iter=1, warm_start=True, eta0=0.03, random_state=42)
error_progress = []
for _ in range(100):
    perceptron.fit(data, labels)
    error_progress.append(np.mean(perceptron.predict(data) != labels))

# Plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

plt.show()
