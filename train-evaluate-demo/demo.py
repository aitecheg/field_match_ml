import numpy as np
import pandas as pd
import tensorflow as tf

from util import ManDist
from util import make_w2v_embeddings
from util import split_and_zero_padding

# File paths
TEST_CSV = './repo/demo.csv'

demo_dataset = []
original_test_df = pd.read_csv(TEST_CSV, keep_default_na=False)
for i, field in enumerate(list(original_test_df["question1"])):
    for j, value in enumerate(list(original_test_df["question2"])):
        demo_dataset.append((field, value, int(i == j)))

fields, vals, label = list(zip(*demo_dataset))

test_df = pd.DataFrame({
    'question1': fields,
    "question2": vals,
    "is_duplicate": label
})

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
    print(prediction.shape)
    for i in range(83):
        arg_max = np.argsort(prediction[i * 83:(i + 1) * 83], axis=0)
        print("filed:", original_test_df.iloc[i]["question1"], "[predicted:", (original_test_df.iloc[arg_max[-1][0]]["question2"], original_test_df.iloc[arg_max[-2][0]]["question2"], original_test_df.iloc[arg_max[-3][0]]["question2"]), "||| actual:", original_test_df.iloc[i]["question2"], "]")

    # print(prediction)
