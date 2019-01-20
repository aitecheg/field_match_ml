"""
  Created by mohammed-alaa
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from util import custom_stacked_bidirectional_GRU_layer, elmo_embeddings, QuoraSequence, ManDistanceLayer

sub_df = pd.read_csv("/home/elsallab/Work/cod/siamese_text/quora/data/sample_submission.csv")
sub_df[["is_duplicate"]] = 0
print(sub_df.shape)
print(sub_df.head())

params_parser = argparse.ArgumentParser(description='my implementation')
params_parser.add_argument('is_elmo', type=int)
params = params_parser.parse_args()

print("Is Elmo", params.is_elmo)

######################################

state_size = 300
staked_layers = 3

is_elmo = params.is_elmo == 1
epochs = 25 if is_elmo else 80
batch_size = 26 if is_elmo else 256
# margin = sqrt(state_size // 2) / 2  # euclidean normalized space
margin = .8  # cosine distance

######################################
if is_elmo:

    with open("/home/elsallab/Work/cod/siamese_text/quora/textual_data.pkl", 'rb') as f:
        _, test_tuples = pickle.load(f)

    test_sequence = QuoraSequence(test_tuples, batch_size, is_demo_generation=False, is_training=False)
else:

    with open("/home/elsallab/Work/cod/siamese_text/quora/vectorized_data.pkl", 'rb') as f:
        _, test_tuples_vectorized = pickle.load(f)

    test_sequence = QuoraSequence(test_tuples_vectorized, batch_size, is_demo_generation=False, is_training=False)

with tf.device('/device:GPU:1'):
    ids = test_sequence.get_id_list()
    siamese = tf.keras.models.load_model('/home/elsallab/Work/cod/siamese_text/quora/models/elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/quora/models/fasttext.h5',

                                         custom_objects={'_custom_stacked_bidirectional_GRU': custom_stacked_bidirectional_GRU_layer(state_size, staked_layers), "elmo_embeddings": elmo_embeddings, "ManDistanceLayer": ManDistanceLayer})
    siamese.summary()

    prediction = siamese.predict_generator(test_sequence, workers=4, use_multiprocessing=True, verbose=2)

    print(np.mean(prediction))
    print(prediction.shape)

    sub_df["is_duplicate"] = sub_df["is_duplicate"].astype("float32")
    prediction = np.squeeze(prediction)
    for index, test_id in enumerate(ids):
        sub_df.at[test_id, "is_duplicate"] = prediction[index]

    print(ids[1461432 - 5:1461432 + 5], "\n", prediction[1461432 - 5:1461432 + 5], "\n", sub_df.iloc[1461432 - 10:1461432 + 5])

    sub_df.to_csv("submission.csv", header=True, index=False)
    # print(prediction)
