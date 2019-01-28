import argparse
import math
import pickle
import shutil

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Embedding, Dense
from tensorflow.python.keras.optimizers import Adam

from util import QuoraSequence, show_metrics, kinitializer, best_saver_callback, custom_stacked_bidirectional_GRU_layer, ManDistanceLayer, CustomElmoEmbeddingLayer

params_parser = argparse.ArgumentParser(description='my implementation')
params_parser.add_argument('is_elmo', type=int)
params_parser.add_argument('is_quora', type=int)
params_parser.add_argument('is_resume', type=int, help="resume training")
params_parser.add_argument('initial_epoch', type=int, help="used to continue(when is_resume=1) training last value reported by tensorboard+1")
params_parser.add_argument('lr', type=float, help="learning rate, for elmo use .00005 for fasttext use .001")

params = params_parser.parse_args()

print("Is Elmo", params.is_elmo)
print("Is quora", params.is_quora)
print("is resume", params.is_resume)
print("initial_epoch", params.initial_epoch)
print("lr", params.lr)
######################################

state_size = 300
staked_layers = 3

is_elmo = params.is_elmo == 1
is_quora = params.is_quora == 1
is_resume = params.is_resume == 1
initial_epoch = params.initial_epoch
lr = params.lr  # .00005 .. .001
if not is_resume:
    print("starting from scratch initial epoch is set to 0")
    initial_epoch = 0

epochs = 20 if is_elmo else 50
batch_size = 16 if is_elmo else 196
# margin = sqrt(state_size // 2) / 2  # euclidean normalized space
margin = .8  # cosine distance

######################################
if not is_resume:
    try:
        if is_elmo:
            shutil.rmtree('/home/elsallab/Work/cod/siamese_text/{}/board/elmo'.format("quora" if is_quora else "repo"))
        else:
            shutil.rmtree('/home/elsallab/Work/cod/siamese_text/{}/board/fast'.format("quora" if is_quora else "repo"))
    except:
        pass

# load data
if is_elmo:
    with open("/home/elsallab/Work/cod/siamese_text/{}/textual_data.pkl".format("quora" if is_quora else "repo"), 'rb') as f:
        main_tuples, _ = pickle.load(f)
else:

    with open("/home/elsallab/Work/cod/siamese_text/{}/quora_lang.pkl".format("quora" if is_quora else "repo"), 'rb') as f:
        embeddings, dictionary, reverse_dictionary = pickle.load(f)

    embedding_dim = embeddings.shape[1]

    with open("/home/elsallab/Work/cod/siamese_text/{}/vectorized_data.pkl".format("quora" if is_quora else "repo"), 'rb') as f:
        main_tuples, _ = pickle.load(f)

if not is_resume:
    train_ids, valid_ids = train_test_split(range(len(main_tuples)), test_size=.05)
    with open("/home/elsallab/Work/cod/siamese_text/{}/data_split.pkl".format("quora" if is_quora else "repo"), 'wb') as f:
        pickle.dump((train_ids, valid_ids), f)
else:
    with open("/home/elsallab/Work/cod/siamese_text/{}/data_split.pkl".format("quora" if is_quora else "repo"), 'rb') as f:
        train_ids, valid_ids = pickle.load(f)

train_split, valid_split = [], []
for id_ in train_ids:
    train_split.append(main_tuples[id_])
for id_ in valid_ids:
    valid_split.append(main_tuples[id_])

train_sequence = QuoraSequence(train_split, batch_size, is_demo_generation=False, do_shuffle=True)
valid_sequence = QuoraSequence(valid_split, batch_size, is_demo_generation=False, do_shuffle=True)
X_valid, Y_valid = valid_sequence.get_entire_list()

with tf.device('/device:GPU:0'):
    # Initialize session
    tokens = Input(shape=(None,), dtype="string" if is_elmo else "float32")
    sequence_length = Input(shape=(1,), dtype="int32")

    # embedding layer for tokens
    if is_elmo:
        # embedded = Lambda(elmo_embeddings)([tokens, sequence_length])
        embedded = CustomElmoEmbeddingLayer()([tokens, sequence_length])

    else:
        embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(None,), trainable=False)(tokens)

with tf.device('/device:GPU:1'):
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
    siamese.compile(loss='mean_squared_error', optimizer=Adam(lr=lr), metrics=['accuracy'])

    siamese.summary()

    if is_resume:
        siamese.load_weights('/home/elsallab/Work/cod/siamese_text/{}/models/elmo.h5'.format("quora" if is_quora else "repo") if is_elmo else '/home/elsallab/Work/cod/siamese_text/{}/models/fasttext.h5'.format("quora" if is_quora else "repo"))
        initial_value_validation_acc = siamese.evaluate(X_valid, batch_size=batch_size, y=Y_valid, verbose=2)[1]  # loss,accuracy
        print("restored model with initial validation accuracy", initial_value_validation_acc)
    else:
        initial_value_validation_acc = -math.inf

    siamese.fit_generator(generator=train_sequence, steps_per_epoch=4000, initial_epoch=initial_epoch,
                          validation_data=valid_sequence, validation_steps=len(valid_sequence),
                          epochs=epochs, use_multiprocessing=False, workers=4, max_queue_size=2
                          , callbacks=[tf.keras.callbacks.TensorBoard(log_dir='/home/elsallab/Work/cod/siamese_text/{}/board/{}/{}'.format("quora" if is_quora else "repo", "elmo" if is_elmo else "fast", initial_epoch)
                                                                      ),
                                       best_saver_callback(siamese, '/home/elsallab/Work/cod/siamese_text/{}/models/elmo.h5'.format("quora" if is_quora else "repo") if is_elmo else '/home/elsallab/Work/cod/siamese_text/{}/models/fasttext.h5'.format("quora" if is_quora else "repo")
                                                           , "val_acc", minimize=False, metric_initial_value=initial_value_validation_acc)], verbose=2, )

    print("Evaluating")
    X_train, Y_train = train_sequence.get_entire_list()
    prediction = siamese.predict(X_train)
    show_metrics(Y_train, prediction)

    print("loss,accuracy", siamese.evaluate(X_valid, batch_size=batch_size, y=Y_valid, verbose=2))
    prediction = siamese.predict(X_valid)
    show_metrics(Y_valid, prediction)
    # ###################################################################################
