import numpy      as np
import tensorflow as tf

from funcy import *


def op(num_blocks, d_model, num_heads, d_ff, x_vocab_size, y_vocab_size, x_maximum_position, y_maximum_position, dropout_rate):
    def dense(units):
        return tf.keras.layers.Dense(units)

    def dropout(rate):
        return tf.keras.layers.Dropout(rate)

    def embedding(input_dim, output_dim):
        return tf.keras.layers.Embedding(input_dim, output_dim)

    def layer_normalization():
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def relu():
        return tf.keras.layers.ReLU()

    def reshape(target_shape):
        return tf.keras.layers.Reshape(target_shape)

    def transpose(perm):
        return func_partial(tf.transpose, perm=perm)

    ####

    def scaled_dot_product_attention(x):
        query, key, value, mask = x

        return tf.matmul(tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32)) + mask * -1e9, axis=-1), value)

    def multi_head_attention(d_model, num_heads):
        split  = rcompose(reshape((-1, num_heads, d_model // num_heads)),
                          transpose((0, 2, 1, 3)))
        concat = rcompose(transpose((0, 2, 1, 3)),
                          reshape((-1, d_model)))

        def op(inputs):
            q, k, v, mask = inputs

            o = scaled_dot_product_attention((split(dense(d_model)(q)),
                                              split(dense(d_model)(k)),
                                              split(dense(d_model)(v)),
                                              mask))
            o = concat(o)
            o = dense(d_model)(o)

            return o

        return op

    def point_wise_feed_forward(d_model, d_ff):
        return rcompose(dense(d_ff),
                        relu(),
                        dense(d_model))

    def encoder_block(d_model, num_heads, d_ff, dropout_rate):
        def op(inputs):
            x, mask = inputs

            o = layer_normalization()(dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((x, x, x, mask))) + x)
            o = layer_normalization()(dropout(dropout_rate)(point_wise_feed_forward(d_model, d_ff)(o))                 + o)

            return o

        return op

    def decoder_block(d_model, num_heads, d_ff, dropout_rate):
        def op(inputs):
            y, y_mask, z, z_mask = inputs

            o = layer_normalization()(dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((y, y, y, y_mask))) + y)
            o = layer_normalization()(dropout(dropout_rate)(multi_head_attention(d_model, num_heads)((o, z, z, z_mask))) + o)
            o = layer_normalization()(dropout(dropout_rate)(point_wise_feed_forward(d_model, d_ff)(o))                   + o)

            return o

        return op

    def get_positional_encoding(maximum_position, d_model):
        result = np.empty((maximum_position, d_model), dtype=np.float32)

        angles = np.arange(maximum_position)[:, np.newaxis] / np.power(10000, 2 * np.arange(d_model // 2) / d_model)

        result[:, 0::2] = np.sin(angles)  # 偶数はsin
        result[:, 1::2] = np.cos(angles)  # 奇数はcos
        result = tf.cast(result[np.newaxis, ...], dtype=tf.float32)

        return result

    def encoder(num_blocks, d_model, num_heads, d_ff, vocab_size, maximum_position, dropout_rate):
        normalize_factor    = tf.math.sqrt(tf.cast(d_model, tf.float32))
        positional_encoding = get_positional_encoding(maximum_position, d_model)

        def op(inputs):
            x, mask = inputs

            o = dropout(dropout_rate)(embedding(vocab_size, d_model)(x) * normalize_factor + positional_encoding[:, :tf.shape(x)[1], :])

            for _ in range(num_blocks):
                o = encoder_block(d_model, num_heads, d_ff, dropout_rate)((o, mask))

            return o

        return op

    def decoder(num_blocks, d_model, num_heads, d_ff, vocab_size, maximum_position, dropout_rate):
        normalize_factor    = tf.math.sqrt(tf.cast(d_model, tf.float32))
        positional_encoding = get_positional_encoding(maximum_position, d_model)

        def op(inputs):
            y, y_mask, z, z_mask = inputs

            o = dropout(dropout_rate)(embedding(vocab_size, d_model)(y) * normalize_factor + positional_encoding[:, :tf.shape(y)[1], :])

            for _ in range(num_blocks):
                o = decoder_block(d_model, num_heads, d_ff, dropout_rate)((o, y_mask, z, z_mask))

            return o

        return op

    def get_padding_mask(x):
        return tf.cast(tf.math.equal(x, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def get_look_ahead_mask(size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    def transformer():
        def op(inputs):
            x, y = inputs

            o = encoder(num_blocks, d_model, num_heads, d_ff, x_vocab_size, x_maximum_position, dropout_rate)((x, get_padding_mask(x)))
            o = decoder(num_blocks, d_model, num_heads, d_ff, y_vocab_size, y_maximum_position, dropout_rate)((y, tf.maximum(get_look_ahead_mask(tf.shape(y)[1]), get_padding_mask(y)), o, get_padding_mask(x)))
            o = dense(y_vocab_size)(o)

            return o

        return op

    return transformer()


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateSchedule, self).__init__()

        self.d_model      = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.d_model ** -0.5 * tf.math.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)


class Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Loss, self).__init__()

        self.sparse_categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, y_true, y_pred):
        return tf.reduce_mean(self.sparse_categorical_crossentropy(y_true, y_pred) * tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32))
