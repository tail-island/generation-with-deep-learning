import matplotlib.pyplot as plot
import numpy             as np
import tensorflow        as tf

from funcy import *


def plot_latent(n):
    decoder = tf.keras.models.load_model('./models/decoder.h5')

    image = np.zeros((28 * n, 28 * n))

    xs, ys = np.meshgrid(np.linspace(-2, 2, n), np.linspace(-2, 2, n))
    images = np.reshape(decoder.predict(np.transpose(np.array((xs.flatten(), ys.flatten())))), (n, n, 28, 28))

    for i in range(n):
        for j in range(n):
            image[i * 28: i * 28 + 28, j * 28: j * 28 + 28] = images[i, j]

    plot.imshow(image, cmap='Greys_r')
    plot.show()


def main():
    plot_latent(30)


if __name__ == '__main__':
    main()
