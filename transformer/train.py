import numpy      as np
import tensorflow as tf

from data_set import create_dataset
from funcy    import *
from params   import *
from model    import LearningRateSchedule, Loss, op


def main():
    np.random.seed(0)

    (train_x, train_y), (valid_x, valid_y) = create_dataset()

    transformer = tf.keras.Model(*juxt(identity, op(NUM_BLOCKS, D_MODEL, NUM_HEADS, D_FF, X_VOCAB_SIZE, Y_VOCAB_SIZE, X_MAXIMUM_POSITION, Y_MAXIMUM_POSITION, DROPOUT_RATE))([tf.keras.Input(shape=(None,)), tf.keras.Input(shape=(None,))]))
    transformer.compile(tf.keras.optimizers.Adam(LearningRateSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss=Loss(), metrics=('accuracy',))
    transformer.fit((train_x, train_y[:, :-1]), train_y[:, 1:], batch_size=64, epochs=100, validation_data=((valid_x, valid_y[:, :-1]), valid_y[:, 1:]))

    transformer.save_weights('./model/transformer_weights')

    # tf.keras.models.save_model(transformer, './model', include_optimizer=False)  # load_modelで、tf.linalg.band_partが失敗しちゃう。2.4で修正済み。


if __name__ == '__main__':
    main()
