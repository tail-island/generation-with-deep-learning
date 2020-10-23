import matplotlib.animation as animation
import matplotlib.pyplot    as plot
import numpy                as np
import tensorflow           as tf

from funcy import *


def load_dataset():
    def normalize(x):
        return np.reshape(x, (np.shape(x)[0], 28, 28, 1)).astype(np.float32) / 255.0

    return map(juxt(compose(normalize, first), second), tf.keras.datasets.mnist.load_data())


def morph(n):
    def get_z_mean(y):
        z = np.array(encoder.predict(test_x[np.where(test_y == y)]))
        return np.mean(z[0, :, :], axis=0)

    encoder = tf.keras.models.load_model('./models/encoder.h5')
    decoder = tf.keras.models.load_model('./models/decoder.h5')

    _, (test_x, test_y) = load_dataset()

    images = np.reshape(decoder.predict(np.transpose(tuple(map(lambda start, end: np.linspace(start, end, n), get_z_mean(1), get_z_mean(9))))), (n, 28, 28))

    artist_animation = animation.ArtistAnimation(plot.figure(), tuple(map(lambda image: (plot.imshow(image, cmap='Greys_r'),), images)), interval=100, repeat_delay=1000)
    plot.show()


def main():
    morph(30)


if __name__ == '__main__':
    main()
