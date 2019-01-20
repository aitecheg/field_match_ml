"""
  Created by mohammed-alaa
"""
import argparse
import pickle

import tensorflow as tf

from util import show_metrics, custom_stacked_bidirectional_GRU_layer, ManDistanceLayer, QuoraSequence, CustomElmoEmbeddingLayer

params_parser = argparse.ArgumentParser(description='my implementation')
params_parser.add_argument('is_elmo', type=int)
params_parser.add_argument('is_quora', type=int)
params = params_parser.parse_args()

print("Is Elmo", params.is_elmo)
print("Is quora", params.is_quora)
######################################

state_size = 300
staked_layers = 3

is_elmo = params.is_elmo == 1
is_quora = params.is_quora == 1

epochs = 15 if is_elmo else 50
batch_size = 26 if is_elmo else 256
# margin = sqrt(state_size // 2) / 2  # euclidean normalized space
margin = .8  # cosine distance

######################################

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

with open("/home/elsallab/Work/cod/siamese_text/{}/data_split.pkl".format("quora" if is_quora else "repo"), 'rb') as f:
    _, valid_ids = pickle.load(f)

    valid_split = []
    for id_ in valid_ids:
        valid_split.append(main_tuples[id_])

    valid_sequence = QuoraSequence(valid_split, batch_size, is_demo_generation=False, do_shuffle=True)
    X_valid, Y_valid = valid_sequence.get_entire_list()

with tf.device('/device:GPU:1'):
    siamese = tf.keras.models.load_model(('/home/elsallab/Work/cod/siamese_text/{}/models/elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/{}/models/fasttext.h5').format("quora" if is_quora else "repo"), custom_objects={'_custom_stacked_bidirectional_GRU': custom_stacked_bidirectional_GRU_layer(state_size, staked_layers), "CustomElmoEmbeddingLayer": CustomElmoEmbeddingLayer, "ManDistanceLayer": ManDistanceLayer})

    siamese.summary()
    # kinitializer()
    # siamese.load_weights(('/home/elsallab/Work/cod/siamese_text/{}/models/elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/{}/models/fasttext.h5').format("quora" if is_quora else "repo"))
    print("loss,accuracy", siamese.evaluate(X_valid, batch_size=batch_size, y=Y_valid, verbose=2))
    prediction = siamese.predict(X_valid)
    show_metrics(Y_valid, prediction)
