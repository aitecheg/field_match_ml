"""
  Created by mohammed-alaa
"""
import argparse
import pickle

import tensorflow as tf

from util import show_metrics, custom_stacked_bidirectional_GRU_layer, elmo_embeddings, ManDistanceLayer

params_parser = argparse.ArgumentParser(description='my implementation')
params_parser.add_argument('is_elmo', type=int)
params_parser.add_argument('is_quora', type=int)
params = params_parser.parse_args()

print("Is Elmo", params.is_elmo)
print("Is quora", params.is_quora)
######################################

state_size = 256
staked_layers = 3

is_elmo = params.is_elmo == 1
is_quora = params.is_quora == 1

epochs = 15 if is_elmo else 50
batch_size = 26 if is_elmo else 256
# margin = sqrt(state_size // 2) / 2  # euclidean normalized space
margin = .8  # cosine distance

######################################
with open("/home/elsallab/Work/cod/siamese_text/{}/valid_data.pkl".format("quora" if is_quora else "repo"), 'rb') as f:
    X_valid, Y_valid = pickle.load(f)

with tf.device('/device:GPU:1'):
    siamese = tf.keras.models.load_model(('/home/elsallab/Work/cod/siamese_text/{}/models/my_elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/{}/models/my_fasttext.h5').format("quora" if is_quora else "repo"),
                                         custom_objects={'_custom_stacked_bidirectional_GRU': custom_stacked_bidirectional_GRU_layer(state_size, staked_layers), "elmo_embeddings": elmo_embeddings, "ManDistanceLayer": ManDistanceLayer})
    siamese.summary()

    print("loss,accuracy", siamese.evaluate(X_valid, batch_size=batch_size, y=Y_valid, verbose=2))
    prediction = siamese.predict(X_valid)
    show_metrics(Y_valid, prediction)
