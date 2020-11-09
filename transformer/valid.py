import numpy      as np
import tensorflow as tf

from data_set import ENCODE, decode, create_dataset
from funcy    import *
from params   import *
from model    import LearningRateSchedule, Loss, op


def translate(transformer, x):
    y = [ENCODE['^']]

    while True:
        y.append(np.argmax(transformer.predict([tf.expand_dims(x, 0), tf.expand_dims(y,  0)])[-1, -1]))

        if y[-1] == ENCODE['$']:
            break

    return np.array(y)


def main():
    np.random.seed(0)

    _, (valid_x, valid_y) = create_dataset()

    transformer = tf.keras.Model(*juxt(identity, op(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, X_VOCAB_SIZE, Y_VOCAB_SIZE, X_MAXIMUM_POSITION, Y_MAXIMUM_POSITION, DROPOUT_RATE))([tf.keras.Input(shape=(None,)), tf.keras.Input(shape=(None,))]))
    transformer.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=Loss(), metrics=('accuracy',))
    transformer.load_weights('./model/transformer_weights')

    # transformer = tf.keras.models.load_model('./model')  # tf.linalg.band_partが失敗しちゃう。2.4で修正済み。

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
