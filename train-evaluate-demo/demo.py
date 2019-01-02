import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from util import ManDist
from util import make_w2v_embeddings
from util import split_and_zero_padding

params_parser = argparse.ArgumentParser(description='repo')
params_parser.add_argument('gpu', type=int)
params_parser.add_argument('demo', type=str)
params = params_parser.parse_args()

# File paths
TEST_CSV = './data/{}.csv'.format(params.demo)

demo_df = pd.read_csv(TEST_CSV, keep_default_na=False)
print(demo_df)
is_true = (demo_df["is_duplicate"] == 1).values

positive_indexes = []
for i, _is_true in enumerate(is_true):
    if _is_true:
        positive_indexes.append(i)
positive_indexes.append(len(is_true))
print("original dataframe size", demo_df.shape)

for q in ['question1', 'question2']:
    demo_df[q + '_n'] = demo_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(demo_df, embedding_dim=embedding_dim, empty_w2v=False)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --
print(np.array(test_df["is_duplicate"]).shape)
print(X_test['left'].shape, X_test['right'].shape)
with tf.device('/device:GPU:{}'.format(params.gpu)):
    model = tf.keras.models.load_model('./models/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
    model.summary()

    print(model.evaluate([X_test['left'], X_test['right']], batch_size=512, y=np.array(test_df["is_duplicate"])))
    prediction = model.predict([X_test['left'], X_test['right']])

    for i in range(len(positive_indexes) - 1):
        predictions = prediction[positive_indexes[i]:positive_indexes[i + 1]]
        predicted_indexes = np.argsort(predictions, axis=0) + positive_indexes[i]
        print(demo_df.iloc[positive_indexes[i]]["question1"], ">>>", demo_df.iloc[positive_indexes[i]]["question2"], "\n"
              , "\n".join(list((demo_df.iloc[predicted_index]["question2"] + ":" + str(predictions[predicted_index - i * (positive_indexes[i + 1] - positive_indexes[i])][0])) for predicted_index in predicted_indexes[::-1, 0])))

    # print(prediction)
