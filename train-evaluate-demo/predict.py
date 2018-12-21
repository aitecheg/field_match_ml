import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from util import ManDist
from util import make_w2v_embeddings
from util import split_and_zero_padding

# File paths
TEST_CSV = './repo/train.csv'

# Load training set
test_df = pd.read_csv(TEST_CSV, keep_default_na=False)
for q in ['question1', 'question2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --
print(np.array(test_df["is_duplicate"]).shape)
print(X_test['left'].shape, X_test['right'].shape)
with tf.device('/device:GPU:1'):
    model = tf.keras.models.load_model('./repo/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
    model.summary()
    print(model.evaluate([X_test['left'], X_test['right']], batch_size=512, y=np.array(test_df["is_duplicate"])))  # [0.1433791877169883, 0.836463501020833] [0.09123201929869311, 0.8700960347451158]

    prediction = model.predict([X_test['left'], X_test['right']])
    print(confusion_matrix(np.array(test_df["is_duplicate"]), prediction < .5))
    print("recall", recall_score(np.array(test_df["is_duplicate"]), prediction < .5))
    print("precision", precision_score(np.array(test_df["is_duplicate"]), prediction < .5))
    # print(prediction)
