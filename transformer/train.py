import numpy      as np
import tensorflow as tf

from data_set import ENCODE, WORDS, decode, create_dataset
from model    import Transformer


NUM_BLOCKS   = 3           # 簡単なタスクなので、Attention is all you needの半分で
D_MODEL      = 256         # 簡単なタスクなので、Attention is all you needの半分で
D_FF         = 1024        # 簡単なタスクなので、Attention is all you needの半分で
NUM_HEADS    = 4           # 簡単なタスクなので、Attention is all you needの半分で
DROPOUT_RATE = 0.1         # ここは、Attention is all you needのまま
X_VOCAB_SIZE = len(WORDS)
Y_VOCAB_SIZE = len(WORDS)  # 出力には演算記号はないのだけど、面倒なので含めます


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(LearningRateSchedule, self).__init__()

    self.d_model      = tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    return self.d_model ** -0.5 * tf.math.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)


sparse_categorical_crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(y_true, y_pred):
    return tf.reduce_mean(sparse_categorical_crossentropy(y_true, y_pred) * tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32))


def translate(transformer, x):
    y = [ENCODE['^']]

    while True:
        y.append(np.argmax(transformer([tf.expand_dims(x, 0), tf.expand_dims(y,  0)])[-1, -1]))

        if y[-1] == ENCODE['$']:
            break

    return np.array(y)


def main():
    np.random.seed(0)

    (train_x, train_y), (valid_x, valid_y) = create_dataset()

    transformer = Transformer(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, X_VOCAB_SIZE, Y_VOCAB_SIZE, X_VOCAB_SIZE, Y_VOCAB_SIZE, DROPOUT_RATE)
    transformer.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=loss_function, metrics=('accuracy',))
    transformer.fit((train_x, train_y[:, :-1]), train_y[:, 1:], batch_size=64, epochs=100, validation_data=((valid_x, valid_y[:, :-1]), valid_y[:, 1:]))

    c = 0

    for x, y in zip(valid_x, valid_y):
        y_pred = translate(transformer, x)

        print('question:   {}'.format(decode(x     ).replace('^', '').replace('$', '')))
        print('answer:     {}'.format(decode(y     ).replace('^', '').replace('$', '')))
        print('prediction: {}'.format(decode(y_pred).replace('^', '').replace('$', '')))

        if np.shape(y_pred) == np.shape(y[y != 0]) and all(y_pred == y[y != 0]):
            c += 1
        else:
            print('NG')

        print()

    print('{:0.3f}'.format(c / len(valid_x)))


if __name__ == '__main__':
    main()
