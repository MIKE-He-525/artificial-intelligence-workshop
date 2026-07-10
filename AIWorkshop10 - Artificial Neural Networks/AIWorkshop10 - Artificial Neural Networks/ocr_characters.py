import numpy as np
from sklearn.neural_network import MLPClassifier

input_file = 'letter.data'
num_datapoints = 50
orig_labels = 'omandig'
num_orig_labels = len(orig_labels)
num_train = int(0.9 * num_datapoints)
num_test = num_datapoints - num_train
start = 6
end = -1

data = []
labels = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        list_vals = line.split('\t')
        if list_vals[1] not in orig_labels:
            continue
        labels.append(orig_labels.index(list_vals[1]))
        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)
        if len(data) >= num_datapoints:
            break

data = np.asarray(data)
labels = np.asarray(labels)

nn = MLPClassifier(
    hidden_layer_sizes=(128, 16),
    activation='relu',
    solver='adam',
    max_iter=1000,
    learning_rate_init=0.001,
    random_state=42,
)
nn.fit(data[:num_train], labels[:num_train])

print('\nTesting on unknown data:')
predicted_test = nn.predict(data[num_train:])
for i in range(num_test):
    true_idx = labels[num_train + i]
    pred_idx = predicted_test[i]
    print('\nOriginal:', orig_labels[true_idx])
    print('Predicted:', orig_labels[pred_idx])
