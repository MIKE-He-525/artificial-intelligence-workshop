import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


def get_data(num_points):
    wave_1 = 0.5 * np.sin(np.arange(0, num_points))
    wave_2 = 3.6 * np.sin(np.arange(0, num_points))
    wave_3 = 1.1 * np.sin(np.arange(0, num_points))
    wave_4 = 4.7 * np.sin(np.arange(0, num_points))

    amp_1 = np.ones(num_points)
    amp_2 = 2.1 + np.zeros(num_points)
    amp_3 = 3.2 * np.ones(num_points)
    amp_4 = 0.8 + np.zeros(num_points)
    wave = np.array([wave_1, wave_2, wave_3, wave_4]).reshape(num_points * 4, 1)
    amp = np.array([[amp_1, amp_2, amp_3, amp_4]]).reshape(num_points * 4, 1)
    return wave, amp


def visualize_output(model, num_points_test):
    wave, amp = get_data(num_points_test)
    output = model.predict(wave).reshape(-1, 1)
    plt.plot(amp.reshape(num_points_test * 4))
    plt.plot(output.reshape(num_points_test * 4))


if __name__ == '__main__':
    num_points = 40
    wave, amp = get_data(num_points)

    # Feedforward NN approximating ELM time-sequential mapping (10 hidden units)
    nn = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation='tanh',
        solver='adam',
        max_iter=1200,
        learning_rate_init=0.005,
        random_state=42,
    )
    nn.fit(wave, amp.ravel())
    error_progress = nn.loss_curve_
    output = nn.predict(wave).reshape(-1, 1)

    plt.subplot(211)
    plt.plot(error_progress)
    plt.xlabel('Number of epochs')
    plt.ylabel('Error (MSE)')

    plt.subplot(212)
    plt.plot(amp.reshape(num_points * 4))
    plt.plot(output.reshape(num_points * 4))
    plt.legend(['Original', 'Predicted'])

    plt.figure()
    plt.subplot(211)
    visualize_output(nn, 82)
    plt.xlim([0, 300])

    plt.subplot(212)
    visualize_output(nn, 49)
    plt.xlim([0, 300])

    plt.show()
