import matplotlib.pyplot as plot
import numpy             as np
import tensorflow        as tf

from funcy     import *
from itertools import product, starmap


def create_generator_op(height, width):
    return rcompose(tf.keras.layers.Dense(height // 2 // 2 * width // 2 // 2 * 256),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),

                    tf.keras.layers.Reshape((height // 2 // 2, width // 2 // 2, 256)),

                    tf.keras.layers.Conv2D(256, 3, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),

                    tf.keras.layers.UpSampling2D(),

                    tf.keras.layers.Conv2D(128, 3, padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU(),

                    tf.keras.layers.UpSampling2D(),

                    tf.keras.layers.Conv2D(1, 3, padding='same'),
                    tf.keras.layers.Activation('tanh'))


def create_discriminator_op():
    return rcompose(tf.keras.layers.Conv2D( 64, 3, strides=2, padding='same'),
                    tf.keras.layers.LeakyReLU(),

                    tf.keras.layers.Conv2D(128, 3, strides=2, padding='same'),
                    tf.keras.layers.LeakyReLU(),

                    tf.keras.layers.Conv2D(256, 3,            padding='same'),
                    tf.keras.layers.LeakyReLU(),

                    tf.keras.layers.Flatten(),

                    tf.keras.layers.Dense(256),
                    tf.keras.layers.LeakyReLU(),

                    tf.keras.layers.Dropout(0.5),

                    tf.keras.layers.Dense(1),
                    tf.keras.layers.Activation('sigmoid'))


Z_DIM      = 100
WIDTH      = 28
HEIGHT     = 28
EPOCH_SIZE = 200
BATCH_SIZE = 32


def load_dataset():
    def normalize(x):
        return (np.reshape(x, (np.shape(x)[0], HEIGHT, WIDTH, 1)).astype(np.float32) - 127.5) / 127.5

    return np.array(tuple(concat(*map(compose(normalize, first), tf.keras.datasets.mnist.load_data()))))


def main():
    def draw_image():
        image = np.zeros((28 * 10, 28 * 10))

        for (i, j), y in zip(product(range(10), range(10)), np.reshape((generator.predict(z) + 1) / 2, (10 * 10, 28, 28))):
            image[i * 28: i * 28 + 28, j * 28: j * 28 + 28] = y

        return image

    def save_image(epoch):
        figure = plot.figure()
        plot.imshow(draw_image(), cmap='Greys_r')
        figure.savefig('images/{:04d}.png'.format(epoch))
        plot.close()

    generator = tf.keras.Model(*juxt(identity, create_generator_op(HEIGHT, WIDTH))(tf.keras.Input(shape=(Z_DIM,))), name='generator')
    generator.summary()

    discriminator = tf.keras.Model(*juxt(identity, create_discriminator_op())(tf.keras.Input(shape=(HEIGHT, WIDTH, 1))), name='discriminator')
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    combined = tf.keras.Model(*juxt(identity, rcompose(generator, discriminator))(tf.keras.Input(shape=(Z_DIM,))), name='combined')
    combined.layers[-1].trainable = False
    combined.summary()
    combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))

    x = load_dataset()
    z = np.random.normal(0, 1, (10 * 10, Z_DIM))

    for epoch in range(EPOCH_SIZE):
        save_image(epoch)

        np.random.shuffle(x)

        for i in range(len(x) // BATCH_SIZE):
            temp_x = x[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE]
            temp_z = np.random.normal(0, 1, (BATCH_SIZE, Z_DIM))

            d_loss_real = discriminator.train_on_batch(temp_x, np.ones(BATCH_SIZE))
            d_loss_fake = discriminator.train_on_batch(generator.predict(temp_z), np.zeros(BATCH_SIZE))
            g_loss      = combined.train_on_batch(temp_z, np.ones(BATCH_SIZE))

            print('epoch: {:04d}, batch: {:04d}, discriminator loss: {:.4f}, generator loss: {:.4f}'.format(epoch, i, (np.add(d_loss_real, d_loss_fake) / 2), g_loss))

    save_image(EPOCH_SIZE)

    generator.save('./models/generator.h5')
    discriminator.save('./models/discriminator.h5')


if __name__ == '__main__':
    main()
