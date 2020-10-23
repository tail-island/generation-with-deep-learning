import numpy      as np
import tensorflow as tf

from funcy import *


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, encoder, decoder, name='variational-auto-encoder', **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        mean, logvar = self.encoder(x)
        z            = mean + tf.exp(0.5 * logvar) * tf.keras.backend.random_normal(shape=tf.shape(mean))  # Reparameterization Trick
        y            = self.decoder(z)

        binary_crossentropy         = tf.keras.losses.binary_crossentropy(x, y) * np.shape(x)[1] * np.shape(x)[2]  # https://blog.keras.io/building-autoencoders-in-keras.htmlで画素数倍していたので、* np.shape()[1] * np.shape()[2]しました
        kullback_leibler_divergence = 0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))  # 正規分布の場合の、カルバック・ライブラー情報量の近似式

        self.add_loss(binary_crossentropy - kullback_leibler_divergence)

        return y


def create_encoder_op(latent_dim):
    return rcompose(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
                    tf.keras.layers.Flatten(),
                    ljuxt(tf.keras.layers.Dense(latent_dim),   # 平均を生成
                          tf.keras.layers.Dense(latent_dim)))  # 分散を生成


def create_decoder_op(height, width):
    return rcompose(tf.keras.layers.Dense(height // 2 // 2 * width // 2 // 2 * 32, activation='relu'),
                    tf.keras.layers.Reshape((height // 2 // 2, width // 2 // 2, 32)),
                    tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation='relu'),
                    tf.keras.layers.Conv2DTranspose( 1, 3, 1, padding='same', activation='sigmoid'))


LATENT_DIM = 2
WIDTH      = 28
HEIGHT     = 28
EPOCH_SIZE = 30


def load_dataset():
    def normalize(x):
        return np.reshape(x, (np.shape(x)[0], HEIGHT, WIDTH, 1)).astype(np.float32) / 255.0

    return map(compose(normalize, first), tf.keras.datasets.mnist.load_data())


def main():
    encoder = tf.keras.Model(*juxt(identity, create_encoder_op(LATENT_DIM))(tf.keras.Input(shape=(HEIGHT, WIDTH, 1))))
    encoder.summary()

    decoder = tf.keras.Model(*juxt(identity, create_decoder_op(HEIGHT, WIDTH))(tf.keras.Input(shape=(LATENT_DIM,))))
    decoder.summary()

    vae = VariationalAutoEncoder(encoder, decoder)
    vae.compile(optimizer='adam')

    train_x, test_x = load_dataset()
    vae.fit(x=train_x, batch_size=64, epochs=EPOCH_SIZE, validation_data=(test_x,))

    encoder.save('./models/encoder.h5')
    decoder.save('./models/decoder.h5')


if __name__ == '__main__':
    main()
