import numpy      as np
import tensorflow as tf

from funcy import *


def scaled_dot_product_attention(q, k, v, mask):
    attention_weight = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32)) + mask * -1e9)

    return tf.matmul(attention_weight, v)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name='multi_head_attention', **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)

        self.dense_q     = tf.keras.layers.Dense(d_model)
        self.dense_k     = tf.keras.layers.Dense(d_model)
        self.dense_v     = tf.keras.layers.Dense(d_model)
        self.split_heads = rcompose(tf.keras.layers.Reshape((-1, num_heads, d_model // num_heads)),
                                    func_partial(tf.transpose, perm=(0, 2, 1, 3)))
        self.attention   = scaled_dot_product_attention
        self.concat      = rcompose(func_partial(tf.transpose, perm=(0, 2, 1, 3)),
                                    tf.keras.layers.Reshape((-1, d_model)))
        self.linear      = tf.keras.layers.Dense(d_model)

    def call(self, q, k, v, mask):
        return self.linear(self.concat(self.attention(self.split_heads(self.dense_q(q)),
                                                      self.split_heads(self.dense_k(k)),
                                                      self.split_heads(self.dense_v(v)),
                                                      mask)))


class PointWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, name='point_wise_feed_forward', **kwargs):
        super(PointWiseFeedForward, self).__init__(name=name, **kwargs)

        self.dense_1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(d_model)

    def call(self, x):
        return self.dense_2(self.dense_1(x))


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, name='encoder_block', **kwargs):
        super(EncoderBlock, self).__init__(name=name, **kwargs)

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout_1            = tf.keras.layers.Dropout(dropout_rate)
        self.add_and_norm_1       = rcompose(tf.keras.layers.Add(),
                                             tf.keras.layers.LayerNormalization(epsilon=1e-6))

        self.feed_forward         = PointWiseFeedForward(d_model, d_ff)
        self.dropout_2            = tf.keras.layers.Dropout(dropout_rate)
        self.add_and_norm_2       = rcompose(tf.keras.layers.Add(),
                                             tf.keras.layers.LayerNormalization(epsilon=1e-6))

    def call(self, x, padding_mask):
        result = self.add_and_norm_1([self.dropout_1(self.multi_head_attention(x, x, x, padding_mask)), x])
        result = self.add_and_norm_2([self.dropout_2(self.feed_forward(result)),                        result])

        return result


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, name='decoder_block', **kwargs):
        super(DecoderBlock, self).__init__(name=name, **kwargs)

        self.multi_head_attention_1 = MultiHeadAttention(d_model, num_heads, name="multi_head_attention_1")
        self.dropout_1              = tf.keras.layers.Dropout(dropout_rate)
        self.add_and_norm_1         = rcompose(tf.keras.layers.Add(),
                                               tf.keras.layers.LayerNormalization(epsilon=1e-6))

        self.multi_head_attention_2 = MultiHeadAttention(d_model, num_heads, name="multi_head_attention_2")
        self.dropout_2              = tf.keras.layers.Dropout(dropout_rate)
        self.add_and_norm_2         = rcompose(tf.keras.layers.Add(),
                                               tf.keras.layers.LayerNormalization(epsilon=1e-6))

        self.feed_forward           = PointWiseFeedForward(d_model, d_ff)
        self.dropout_3              = tf.keras.layers.Dropout(dropout_rate)
        self.add_and_norm_3         = rcompose(tf.keras.layers.Add(),
                                               tf.keras.layers.LayerNormalization(epsilon=1e-6))

    def call(self, y, z, look_ahead_mask, padding_mask):
        result = self.add_and_norm_1([self.dropout_1(self.multi_head_attention_1(y,      y, y, look_ahead_mask)), y])
        result = self.add_and_norm_2([self.dropout_2(self.multi_head_attention_2(result, z, z, padding_mask)),    result])
        result = self.add_and_norm_3([self.dropout_3(self.feed_forward(result)),                                  result])

        return result


def get_positional_encoding(maximum_position, d_model):
    result = np.empty((maximum_position, d_model), dtype=np.float32)

    angles = np.arange(maximum_position)[:, np.newaxis] / np.power(10000, 2 * np.arange(d_model // 2) / d_model)

    result[:, 0::2] = np.sin(angles)  # 偶数はsin
    result[:, 1::2] = np.cos(angles)  # 奇数はcos
    result = tf.cast(result[np.newaxis, ...], dtype=tf.float32)

    return result


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, d_ff, vocab_size, maximum_position, dropout_rate=0.1, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.embedding           = tf.keras.layers.Embedding(vocab_size, d_model)
        self.normalize_factor    = tf.math.sqrt(tf.cast(d_model, tf.float32))
        self.positional_encoding = get_positional_encoding(maximum_position, d_model)
        self.dropout             = tf.keras.layers.Dropout(dropout_rate)
        self.blocks              = tuple(map(lambda i: EncoderBlock(d_model, num_heads, d_ff, dropout_rate, name='encoder_block_{}'.format(i)), range(num_blocks)))

    def call(self, x, padding_mask):
        result = self.dropout(self.embedding(x) * self.normalize_factor + self.positional_encoding[:, :np.shape(x)[1], :])

        for block in self.blocks:
            result = block(result, padding_mask)

        return result


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_blocks, d_model, num_heads, d_ff, vocab_size, maximum_position, dropout_rate=0.1, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.embedding           = tf.keras.layers.Embedding(vocab_size, d_model)
        self.normalize_factor    = tf.math.sqrt(tf.cast(d_model, tf.float32))
        self.positional_encoding = get_positional_encoding(maximum_position, d_model)
        self.dropout             = tf.keras.layers.Dropout(dropout_rate)
        self.blocks              = tuple(map(lambda i: DecoderBlock(d_model, num_heads, d_ff, dropout_rate, name='decoder_block_{}'.format(i)), range(num_blocks)))

    def call(self, x, z, combined_mask, padding_mask):
        result = self.dropout(self.embedding(x) * self.normalize_factor + self.positional_encoding[:, :np.shape(x)[1], :])

        for block in self.blocks:
            result = block(result, z, combined_mask, padding_mask)

        return result


def get_padding_mask(x):
    return tf.cast(tf.math.equal(x, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def get_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), tf.constant(-1, dtype=tf.int32), tf.constant(0, dtype=tf.int32))


class Transformer(tf.keras.Model):
    def __init__(self, num_blocks, d_model, num_heads, d_ff, x_vocab_size, y_vocab_size, x_maximum_position, y_maximum_position, dropout_rate=0.1, name='transformer', **kwargs):
        super(Transformer, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(num_blocks, d_model, num_heads, d_ff, x_vocab_size, x_maximum_position, dropout_rate)
        self.decoder = Decoder(num_blocks, d_model, num_heads, d_ff, y_vocab_size, y_maximum_position, dropout_rate)
        self.linear  = tf.keras.layers.Dense(y_vocab_size)

    def call(self, inputs):
        x, y = inputs

        return self.linear(self.decoder(y, self.encoder(x, get_padding_mask(x)), tf.maximum(get_look_ahead_mask(tf.shape(y)[1]), get_padding_mask(y)), get_padding_mask(x)))
