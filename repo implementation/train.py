from time import time

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM

from util import load_embedding_and_vectorize, show_metrics
from util import split_and_zero_padding
from util import ManDist

import argparse

params_parser = argparse.ArgumentParser(description='repo')
params_parser.add_argument('gpu', type=int)
params_parser.add_argument('file', type=str)
params = params_parser.parse_args()

# File paths
TRAIN_CSV = './data/{}.csv'.format(params.file)

# Load training set
train_df = pd.read_csv(TRAIN_CSV, keep_default_na=False)

for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# Make word2vec embeddings
embedding_dim = 250
max_seq_length = 20
use_w2v = True

train_df, embeddings = load_embedding_and_vectorize(train_df)

# Split to train validation
validation_size = int(len(train_df) * 0.05)
training_size = len(train_df) - validation_size

X = train_df[["question1", "question2", 'question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

valid_split = pd.DataFrame({"question1": X_validation["question1"], "question2": X_validation["question2"], "is_duplicate": Y_validation})
valid_split.to_csv("./data/valid_split.csv")

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

with tf.device('/device:GPU:{}'.format(params.gpu)):
    # --

    # Model variables
    gpus = 1
    batch_size = 512 * gpus
    n_epoch = 25
    n_hidden = 64

    ######################################################################################################
    # Define the shared model
    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

    # LSTM
    x.add(LSTM(n_hidden, return_sequences=True))
    x.add(LSTM(n_hidden, return_sequences=True))
    x.add(LSTM(n_hidden))
    shared_model = x

    # The visible layer
    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    # ######################################################################################################
    # # Define the shared model
    # embedding_layer = Embedding(len(embeddings), embedding_dim,
    #                             weights=[embeddings], input_shape=(max_seq_length,), trainable=False)
    #
    # # LSTM
    # x = Sequential()
    # x.add(LSTM(n_hidden, return_sequences=True, input_shape=(max_seq_length, embedding_dim)))
    # x.add(LSTM(n_hidden, return_sequences=True))
    # x.add(LSTM(n_hidden))
    # right_model = x
    #
    # x = Sequential()
    # x.add(LSTM(n_hidden, return_sequences=True, input_shape=(max_seq_length, embedding_dim)))
    # x.add(LSTM(n_hidden, return_sequences=True))
    # x.add(LSTM(n_hidden))
    # left_model = x
    #
    # # The visible layer
    # left_input = Input(shape=(max_seq_length,), dtype='int32')
    # right_input = Input(shape=(max_seq_length,), dtype='int32')
    #
    # left_embedded = embedding_layer(left_input)
    # right_embedded = embedding_layer(right_input)
    #
    # # Pack it all up into a Manhattan Distance model
    # malstm_distance = ManDist()([left_model(left_embedded), right_model(right_embedded)])
    # ######################################################################################################
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    if gpus >= 2:
        # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    # Start trainings
    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))

    # prediction = model.predict([X_train['left'], X_train['right']])
    # show_metrics(Y_train, prediction)
    print("loss,accuracy", model.evaluate([X_validation['left'], X_validation['right']], batch_size=512, y=Y_validation))
    prediction = model.predict([X_validation['left'], X_validation['right']])
    show_metrics(Y_validation, prediction)

    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                            training_end_time - training_start_time))

    model.save('./models/SiameseLSTM.h5')

    # Plot accuracy
    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('./models/history-graph.png')

    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")
