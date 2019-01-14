import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

from util import ManDist, show_metrics, load_embedding_and_vectorize
from util import split_and_zero_padding

params_parser = argparse.ArgumentParser(description='repo')
params_parser.add_argument('gpu', type=int)
params_parser.add_argument('file', type=str)
params = params_parser.parse_args()

# File paths
TEST_CSV = './data/{}.csv'.format(params.file)

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
labels = np.array(test_df["is_duplicate"])
print(labels.shape)
print(X_test['left'].shape, X_test['right'].shape)
with tf.device('/device:GPU:{}'.format(params.gpu)):
    model = tf.keras.models.load_model('./models/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
    model.summary()

    print("loss,accuracy", model.evaluate([X_test['left'], X_test['right']], batch_size=512, y=labels))
    prediction = model.predict([X_test['left'], X_test['right']])
    show_metrics(labels, prediction)
    # print(prediction)
