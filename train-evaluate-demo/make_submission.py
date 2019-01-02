"""
  Created by mohammed-alaa
"""

import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from util import ManDist, load_embedding_and_vectorize
from util import split_and_zero_padding

params_parser = argparse.ArgumentParser(description='repo')
params_parser.add_argument('gpu', type=int)

params = params_parser.parse_args()

# File paths
TEST_CSV = './data/test.csv'

# Load training set
test_df = pd.read_csv(TEST_CSV, keep_default_na=False)
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
max_seq_length = 20
test_df, _ = load_embedding_and_vectorize(test_df)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --
print(X_test['left'].shape, X_test['right'].shape)
with tf.device('/device:GPU:{}'.format(params.gpu)):
    model = tf.keras.models.load_model('./models/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
    model.summary()

    prediction = model.predict([X_test['left'], X_test['right']])
    print(np.mean(prediction))
    print(prediction.shape)

    submission = pd.DataFrame({"test_id": list(range(len(prediction))),
                               "is_duplicate": np.squeeze(prediction)
                               },)

    submission[['test_id', 'is_duplicate']].to_csv("submission.csv", header=True, index=False)
    # print(prediction)
