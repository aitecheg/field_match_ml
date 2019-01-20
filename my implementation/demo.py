import argparse

import numpy as np
import pandas as pd
import prettytable
import tensorflow as tf
from tensorflow.python.keras.engine.saving import model_from_json

from util import vectorize_demo_data, custom_stacked_bidirectional_GRU_layer, CustomElmoEmbeddingLayer, ManDistanceLayer, kinitializer

params_parser = argparse.ArgumentParser(description='my implementation')
params_parser.add_argument('is_elmo', type=int)
params_parser.add_argument('demo', type=str)
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

# File paths
TEST_CSV = '/home/elsallab/Work/cod/siamese_text/repo/data/{}.csv'.format(params.demo)

demo_df = pd.read_csv(TEST_CSV, keep_default_na=False)
print(demo_df)
is_true = (demo_df["is_duplicate"] == 1).values

positive_indexes = []
for i, _is_true in enumerate(is_true):
    if _is_true:
        positive_indexes.append(i)
positive_indexes.append(len(is_true))
print("original dataframe size", demo_df.shape)

feature1, feature2, feature3, feature4, label = vectorize_demo_data(demo_df, is_elmo)

# --
print(label.shape)
print(feature1.shape, feature2.shape, feature3.shape, feature4.shape)
# print(feature1[:3], feature2[:3], feature3[:3], feature4[:3])
with tf.device('/device:GPU:0'):
    siamese = tf.keras.models.load_model('/home/elsallab/Work/cod/siamese_text/repo/models/elmo.h5' if is_elmo else '/home/elsallab/Work/cod/siamese_text/repo/models/fasttext.h5',
                                         custom_objects={'_custom_stacked_bidirectional_GRU': custom_stacked_bidirectional_GRU_layer(state_size, staked_layers), "CustomElmoEmbeddingLayer": CustomElmoEmbeddingLayer, "ManDistanceLayer": ManDistanceLayer})

    # siamese.save_weights('/home/elsallab/Work/cod/siamese_text/repo/models/w8s.h5')
    # open('/home/elsallab/Work/cod/siamese_text/repo/models/w8s.arch', "w").write(siamese.to_json())

    # print("loss,acc", siamese.evaluate([feature1, feature2, feature3, feature4], batch_size=batch_size, y=np.array(label)))

    prediction = siamese.predict([feature1, feature2, feature3, feature4])

    for i in range(len(positive_indexes) - 1):
        predictions = prediction[positive_indexes[i]:positive_indexes[i + 1]]
        predicted_indexes = np.argsort(predictions, axis=0) + positive_indexes[i]
        predictions_act = prettytable.PrettyTable(["field", "actual value"])
        predictions_act.add_row([demo_df.iloc[positive_indexes[i]]["question1"], demo_df.iloc[positive_indexes[i]]["question2"]])
        print(predictions_act)
        predictions_pt = prettytable.PrettyTable(["prediction", "prediction score"])
        for predicted_index in predicted_indexes[::-1, 0]:
            predictions_pt.add_row([demo_df.iloc[predicted_index]["question2"], str(predictions[predicted_index - i * (positive_indexes[i + 1] - positive_indexes[i])][0])])
        print(predictions_pt)
    # print(prediction)
