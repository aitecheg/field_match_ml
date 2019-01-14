import argparse
import pickle
import shutil

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Embedding, Lambda, Dense
from tensorflow.python.keras.optimizers import Adam

from util import QuoraSequence, show_metrics, kinitializer, best_saver_callback, elmo_embeddings, custom_stacked_bidirectional_GRU_layer, ManDistanceLayer

params_parser = argparse.ArgumentParser(description='my implementation')
params_parser.add_argument('is_elmo', type=int)
params = params_parser.parse_args()

print("Is Elmo", params.is_elmo)

######################################

state_size = 300
staked_layers = 3

is_elmo = params.is_elmo == 1
epochs = 15 if is_elmo else 50
batch_size = 16 if is_elmo else 196
# margin = sqrt(state_size // 2) / 2  # euclidean normalized space
margin = .8  # cosine distance

######################################
try:
    if is_elmo:
        shutil.rmtree('/home/elsallab/Work/cod/siamese_text/quora/board/elmo')
    else:
        shutil.rmtree('/home/elsallab/Work/cod/siamese_text/quora/board/fast')
except:
    pass

if is_elmo:

    with open("/home/elsallab/Work/cod/siamese_text/quora/textual_data.pkl", 'rb') as f:
        train_tuples, test_tuples = pickle.load(f)

    train_tuples, valid_tuples, _, _ = train_test_split(train_tuples, range(len(train_tuples)), test_size=.05)
    train_sequence = QuoraSequence(train_tuples, batch_size, is_demo_generation=False, do_shuffle=True)
    valid_sequence = QuoraSequence(valid_tuples, batch_size, is_demo_generation=False, do_shuffle=True)
    test_sequence = QuoraSequence(test_tuples, batch_size, is_demo_generation=False, is_training=False)
else:

    with open("/home/elsallab/Work/cod/siamese_text/quora/quora_lang.pkl", 'rb') as f:
        embeddings, dictionary, reverse_dictionary = pickle.load(f)

    embedding_dim = embeddings.shape[1]

    with open("/home/elsallab/Work/cod/siamese_text/quora/vectorized_data.pkl", 'rb') as f:
        train_tuples_vectorized, test_tuples_vectorized = pickle.load(f)

    train_tuples_vectorized, valid_tuples_vectorized, _, _ = train_test_split(train_tuples_vectorized, range(len(train_tuples_vectorized)), test_size=.05)
    train_sequence = QuoraSequence(train_tuples_vectorized, batch_size, is_demo_generation=False, do_shuffle=True)
    valid_sequence = QuoraSequence(valid_tuples_vectorized, batch_size, is_demo_generation=False, do_shuffle=True)
    test_sequence = QuoraSequence(test_tuples_vectorized, batch_size, is_demo_generation=False, is_training=False)

with open("/home/elsallab/Work/cod/siamese_text/quora/valid_data.pkl", 'wb') as f:
    X_valid, Y_valid = valid_sequence.get_entire_list()
    pickle.dump((X_valid, Y_valid), f)

with tf.device('/device:GPU:1'):
    # Initialize session
    tokens = Input(shape=(None,), dtype="string" if is_elmo else "float32")
    sequence_length = Input(shape=(1,), dtype="int32")

    # embedding layer for tokens
    if is_elmo:
        embedded = Lambda(elmo_embeddings)([tokens, sequence_length])
    else:
        embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(None,), trainable=False)(tokens)

    # bidirectional RNN layer
    final_state = custom_stacked_bidirectional_GRU_layer(state_size, staked_layers)([embedded, sequence_length])
    final_state = Dense(state_size // 2)(final_state)
    shared_model = Model(inputs=[tokens, sequence_length], outputs=final_state)

    # The visible layer
    left_tokens = Input(shape=(None,), dtype="string" if is_elmo else "float32")  # dynamic-length sequence
    right_tokens = Input(shape=(None,), dtype="string" if is_elmo else "float32")  # dynamic-length sequence

    left_feature_length = Input(shape=(1,), dtype="int32")  # corresponding length
    right_feature_length = Input(shape=(1,), dtype="int32")  # corresponding length

    left_output = shared_model([left_tokens, left_feature_length])
    right_output = shared_model([right_tokens, right_feature_length])
    # ###################################################################################
    # cosine_distance = Lambda(cosine_distance_lambda_layer)([left_output, right_output])
    # siamese = Model(inputs=[left_tokens, left_feature_length, right_tokens, right_feature_length], outputs=cosine_distance)
    # siamese.compile(loss=siamese_loss, optimizer=Adam(lr=10e-5))
    # ###################################################################################
    man_distance = [ManDistanceLayer()([left_output, right_output])]
    siamese = Model(inputs=[left_tokens, left_feature_length, right_tokens, right_feature_length], outputs=man_distance)
    kinitializer()
    siamese.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['accuracy'])

    siamese.summary()

    # siamese.load_weights('/home/elsallab/Work/cod/siamese_text/quora/models/my_elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/quora/models/my_fasttext.h5')
    siamese.fit_generator(generator=train_sequence, steps_per_epoch=len(train_sequence),
                          validation_data=valid_sequence, validation_steps=len(valid_sequence),
                          epochs=epochs, use_multiprocessing=False, workers=4, max_queue_size=2
                          , callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/home/elsallab/Work/cod/siamese_text/quora/board/{}'.format("elmo" if is_elmo else "fast"))
            , best_saver_callback(siamese,
                                  '/home/elsallab/Work/cod/siamese_text/quora/models/my_elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/quora/models/my_fasttext.h5'
                                  , "val_acc", minimize=False)], verbose=2, )

    print("Evaluating")
    X_train, Y_train = train_sequence.get_entire_list()
    prediction = siamese.predict(X_train)
    show_metrics(Y_train, prediction)

    print("loss,accuracy", siamese.evaluate(X_valid, batch_size=batch_size, y=Y_valid, verbose=2))
    prediction = siamese.predict(X_valid)
    show_metrics(Y_valid, prediction)
    # ###################################################################################
