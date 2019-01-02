import itertools
import pickle
import string

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from tensorflow.keras import backend
from tensorflow.keras import metrics
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def tokenize_sentence(sentence):
    """Tokenize the sentence"""
    tokens = []
    last = 0

    sentence = sentence.strip()
    for index, char in enumerate(sentence):
        if char not in string.ascii_letters + "0123456789":
            if sentence[last:index]:
                tokens.append(sentence[last:index])
            if sentence[index].strip():
                tokens.append(sentence[index])
            last = index + 1
    if sentence[last:].strip():
        tokens.append(sentence[last:])

    return tokens


def load_embedding_and_vectorize(df, ):
    with open("./models/Quora-Question-Pairs.ft", 'rb') as f:
        vocab_dict, reverse_dict, embeddings = pickle.load(f)

    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 10000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for question in ['question1', 'question2']:

            q2n = []  # q2n -> question numbers representation
            for word in tokenize_sentence(row[question]):
                # If you have never seen a word, append it to vocab dictionary.
                if word in vocab_dict:
                    q2n.append(vocab_dict[word])
                else:
                    q2n.append(1)  # 1 : unknown word

            # Append question as number representation
            df.at[index, question + '_n'] = q2n

    return df, embeddings


def split_and_zero_padding(df, max_seq_length):
    # Split to dicts
    X = {'left': df['question1_n'], 'right': df['question2_n']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


#  --

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def show_metrics(Y, prediction):
    Y = np.squeeze(Y)
    prediction = np.squeeze(prediction)

    print(confusion_matrix(Y, prediction > .5))
    print("recall", recall_score(Y, prediction > .5))
    print("precision", precision_score(Y, prediction > .5))

    # keras placeholder used for evaluation
    video_level_labels_k = backend.placeholder([None], dtype=tf.float32)
    video_level_preds_k = backend.placeholder([None], dtype=tf.float32)

    val_loss_op = backend.mean(metrics.binary_crossentropy(video_level_labels_k, video_level_preds_k))

    video_level_loss, = backend.get_session().run(
        [val_loss_op], feed_dict={video_level_labels_k: Y, video_level_preds_k: prediction})
    print("log loss", video_level_loss)

    # video_level_loss, = backend.get_session().run(
    #     [val_loss_op], feed_dict={video_level_labels_k: Y, video_level_preds_k: np.ones_like(prediction) * 0.36919785302629282})
    #
    # print("log loss", video_level_loss)
