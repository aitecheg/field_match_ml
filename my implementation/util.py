"""
  Created by mohammed-alaa
"""

import math
import os
import pickle
import string
from functools import partial
from random import shuffle

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix, recall_score, precision_score, classification_report
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import metrics
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell

PADDING_TOKEN = "þ"
NONE_TOKEN = "ø"

def tokenize_sentence(sentence):
    """Tokenize the sentence"""
    tokens = []
    last = 0

    sentence = sentence.strip().lower()
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


def tokenize_sentences_list(sentences_list, vocab_set=None):
    """Tokenize list of sentences"""

    tokenized_list = list(map(tokenize_sentence, list(sentences_list)))

    # from multiprocessing.pool import Pool
    # import multiprocessing
    # workers = Pool(multiprocessing.cpu_count())
    # tokenized_list = list(workers.imap(tokenize_sentence, list(sentences_list), chunksize=multiprocessing.cpu_count()))
    # workers.close()
    # workers.join()

    if vocab_set is not None:
        for tokenized_sentence in tokenized_list:
            for word in tokenized_sentence:
                vocab_set.add(word)
    return tokenized_list


def vectorize_tokenized_sentence(sentence, dictionary):
    """from words to indexes for a sentence"""
    transformed_sentence = []
    for word in sentence:
        if word in dictionary:
            transformed_sentence.append(dictionary[word])
        else:
            transformed_sentence.append(dictionary["<unk>"])  # <unk> = 0

    return transformed_sentence


def vectorize_tokenized_sentences(sentences, dictionary):
    return list(map(partial(vectorize_tokenized_sentence, dictionary=dictionary), list(sentences)))


def devectorize_sentence(sentence_vectorized, reverse_dictionary):
    sentence_str = []
    for word_id in sentence_vectorized:
        sentence_str.append(reverse_dictionary[word_id])
    return sentence_str


def sentence_demo(sample, reverse_dictionary=None, is_test=False):
    if not is_test:
        q1_list, q1_len, q2_list, q2_len, label = sample
    else:
        id_, q1_list, q1_len, q2_list, q2_len = sample

    if reverse_dictionary is not None:
        q1_str = "|".join(devectorize_sentence(q1_list, reverse_dictionary))
        q2_str = "|".join(devectorize_sentence(q2_list, reverse_dictionary))
    else:
        q1_str = "|".join(q1_list)
        q2_str = "|".join(q2_list)

    if not is_test:
        print("{}/{}".format(q1_len, len(q1_list)), q1_str, "\n", "{}/{}".format(q2_len, len(q2_list)), q2_str, "same" if label == 1 else "different")
    else:
        print(id_, "===", "{}/{}".format(q1_len, len(q1_list)), q1_str, "\n", "{}/{}".format(q2_len, len(q2_list)), q2_str)

    print("=" * 30)


def pad_sequences_wrapped(sequence_list):
    is_str = type(sequence_list[0][0]) == str
    return pad_sequences(sequence_list, padding="post", truncating="post", dtype=object if is_str else np.int32, value=PADDING_TOKEN if is_str else 0.)


class QuoraSequence(keras.utils.Sequence):
    def __init__(self, data_tuples, batch_size, is_training=True, is_demo_generation=True, do_shuffle=False):
        """get data structure to load data"""
        # each training tuple is :(list of word ids,list of word ids,label)
        # each testing tuple is :(id,list of word ids,list of word ids)
        self.data_source = data_tuples
        self.batch_size = batch_size
        self.is_training = is_training
        self.do_shuffle = do_shuffle
        self.on_epoch_end()
        self.is_demo_generation = is_demo_generation

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # ceiling div

    def get_actual_length(self):
        """Denotes the total data_to_load of samples"""
        return len(self.data_source)

    def shuffle(self):
        if self.do_shuffle:
            shuffle(self.data_source)

    def __getitem__(self, batch_index):
        """Gets one batch"""
        batch = self.data_source[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        if self.is_training:
            q1_list, q2_list, label = list(zip(*batch))
            if self.is_demo_generation:
                ret = pad_sequences_wrapped(q1_list), \
                      np.array(list(map(lambda item: len(item), q1_list))), \
                      pad_sequences_wrapped(q2_list), \
                      np.array(list(map(lambda item: len(item), q2_list))), \
                      np.array(label)
            else:
                ret = [pad_sequences_wrapped(q1_list),
                       np.array(list(map(lambda item: len(item), q1_list))),
                       pad_sequences_wrapped(q2_list),
                       np.array(list(map(lambda item: len(item), q2_list))), ], \
                      np.array(label)
        else:
            ids, q1_list, q2_list = list(zip(*batch))
            if self.is_demo_generation:
                ret = ids, \
                      pad_sequences_wrapped(q1_list), \
                      np.array(list(map(lambda item: len(item), q1_list))), \
                      pad_sequences_wrapped(q2_list), \
                      np.array(list(map(lambda item: len(item), q2_list)))
            else:
                ret = [pad_sequences_wrapped(q1_list),
                       np.array(list(map(lambda item: len(item), q1_list))),
                       pad_sequences_wrapped(q2_list),
                       np.array(list(map(lambda item: len(item), q2_list)))]

        return ret

    def get_entire_list(self):
        if self.is_training:
            q1_list, q2_list, label = list(zip(*self.data_source))
            return [pad_sequences_wrapped(q1_list),
                    np.array(list(map(lambda item: len(item), q1_list))),
                    pad_sequences_wrapped(q2_list),
                    np.array(list(map(lambda item: len(item), q2_list))), ], \
                   np.array(label)
        else:
            ids, q1_list, q2_list = list(zip(*self.data_source))
            return [pad_sequences_wrapped(q1_list),
                    np.array(list(map(lambda item: len(item), q1_list))),
                    pad_sequences_wrapped(q2_list),
                    np.array(list(map(lambda item: len(item), q2_list)))]

    def on_epoch_end(self):
        self.shuffle()


def kinitializer():
    sess = K.get_session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    sess.run(tf.local_variables_initializer())


def show_metrics(Y, prediction):
    Y = np.squeeze(Y)
    prediction = np.squeeze(prediction)

    print(confusion_matrix(Y, prediction > .5))
    print("recall", recall_score(Y, prediction > .5))
    print("precision", precision_score(Y, prediction > .5))
    classification_report(Y, prediction > .5)

    # keras placeholder used for evaluation
    labels_k = backend.placeholder([None], dtype=tf.float32)
    preds_k = backend.placeholder([None], dtype=tf.float32)

    val_loss_op = backend.mean(metrics.binary_crossentropy(labels_k, preds_k))

    loss, = backend.get_session().run(
        [val_loss_op], feed_dict={labels_k: Y, preds_k: prediction})
    print("log loss", loss)
    # loss, = backend.get_session().run(
    #     [val_loss_op], feed_dict={labels_k: Y, preds_k: np.ones_like(prediction) * 0.36919785302629282})
    #
    # print("log loss", loss)


# vectorize_tokenized_sentence(tokenize_sentence("this is a sentence ain't? time 15-42/2"),dictionary)


def best_saver_callback(model, hpy_path, metric_name, minimize, frequency=1):
    """
    :param model: the model being trained
    :param hpy_path: path to save
    :param metric_name: string like "acc" in logs
    :param minimize: bool
    :param frequency: every n epochs
    :return:
    """

    class SaverCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.metric_best_value = (math.inf if minimize else -math.inf)

        def on_epoch_end(self, epoch, logs=None):
            epoch_one_based = epoch + 1
            metric_value = logs[metric_name]
            if epoch_one_based % frequency == 0:

                if minimize and metric_value < self.metric_best_value or not minimize and metric_value > self.metric_best_value:
                    try:
                        os.remove(hpy_path)
                    except:
                        pass
                    model.save(hpy_path)
                    self.metric_best_value = metric_value
                    print("\n", "=" * 50, "\n", "Epoch", epoch_one_based, "Established new baseline for ({}):".format(metric_name), "{0:.5f}".format(metric_value), "\n", "=" * 50, "\n")
                else:
                    print("\n", "=" * 50, "\n", "Epoch", epoch_one_based, "for ({})".format(metric_name), "Baseline:", "{0:.5f}".format(self.metric_best_value), "but got:", "{0:.5f}".format(metric_value), "\n", "=" * 50, "\n")

    return SaverCallback()  # returns callback instance to be consumed by keras


def elmo_embeddings(tokenized):
    # this can't be trainable, lambda = stateless
    tokens_input, tokens_length = tokenized

    elmo_module = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)

    return elmo_module(inputs={
        "tokens": tokens_input,
        "sequence_len": tokens_length[:, 0]
    },
        as_dict=True,
        signature='tokens',
    )['elmo']  # [batch_size, max_length, 1024]


def cosine_distance_lambda_layer(siamese_outputs):
    _sister1_output, _sister2_output = siamese_outputs
    _sister1_output = _sister1_output / (K.sqrt(K.sum(K.pow(_sister1_output, 2), axis=-1, keepdims=True)) + 1e-11)  # l2 normalization
    _sister2_output = _sister2_output / (K.sqrt(K.sum(K.pow(_sister2_output, 2), axis=-1, keepdims=True)) + 1e-11)  # l2 normalization

    return K.sum(_sister1_output * _sister2_output, axis=1, keepdims=True)  # element-wise multiplication..sum


#  --

class ManDistanceLayer(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDistanceLayer, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDistanceLayer, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def L2_distance_layer(siamese_outputs):
    _sister1_output, _sister2_output = siamese_outputs
    return K.sum(K.square(_sister1_output - _sister2_output), axis=1, keepdims=True)  #


def custom_stacked_bidirectional_GRU_layer(state_size, staked_layers):
    class _custom_stacked_bidirectional_GRU(Layer):
        """
        Keras Custom Layer that calculates Manhattan Distance.
        """

        # initialize the layer, No need to include inputs parameter!
        def __init__(self, state_size=state_size, staked_layers=staked_layers, **kwargs):
            self.final_state = None
            self.fw_state_tuple_multicell = None
            self.bw_state_tuple_multicell = None
            self.state_size = state_size
            self.staked_layers = staked_layers
            self.registered_weights = False
            super(_custom_stacked_bidirectional_GRU, self).__init__(**kwargs)

        # input_shape will automatic collect input shapes to build layer
        def build(self, input_shape):
            """
            this is where you will define your weights.
            you also have to add the weights manually
            This method must set self.built = True at the end, which can be done by calling super([Layer], self).build().

            if you leave this with no added internal state use lambda function instead
            """

            self.fw_state_tuple_multicell = MultiRNNCell([tf.contrib.rnn.GRUCell(self.state_size) for _ in range(self.staked_layers)], state_is_tuple=False)
            self.bw_state_tuple_multicell = MultiRNNCell([tf.contrib.rnn.GRUCell(self.state_size) for _ in range(self.staked_layers)], state_is_tuple=False)

            # self.fw_state_tuple_multicell_d = DropoutWrapper((self.fw_state_tuple_multicell), output_keep_prob=.8, state_keep_prob=.8)
            # self.bw_state_tuple_multicell_d = DropoutWrapper((self.bw_state_tuple_multicell), output_keep_prob=.8, state_keep_prob=.8)

            # not yet built
            # self._trainable_weights += ??
            # self._non_trainable_weights += ??

            super(_custom_stacked_bidirectional_GRU, self).build(input_shape)

        # This is where the layer's logic lives.
        def call(self, args, **kwargs):
            """
            this is where the layer's logic lives. Unless you want your layer to support masking,
             you only have to care about the first argument passed to call: the input tensor.
            """
            inputs, sequence_length = args

            _, (final_state_fw, final_state_bw) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_state_tuple_multicell,
                                                cell_bw=self.bw_state_tuple_multicell,
                                                inputs=inputs,
                                                sequence_length=sequence_length[:, 0],
                                                dtype=tf.float32,
                                                )
            if not self.registered_weights:
                self._trainable_weights += self.fw_state_tuple_multicell.trainable_weights
                self._trainable_weights += self.bw_state_tuple_multicell.trainable_weights

                self.registered_weights = True

            self.final_state = K.concatenate([final_state_fw, final_state_bw])

            return self.final_state

        # return output shape
        def compute_output_shape(self, input_shape):
            """in case your layer modifies the shape of its input, you should sapecify here the shape transformation logic.
             This allows Keras to do automatic shape inference."""
            return K.int_shape(self.final_state)

    return _custom_stacked_bidirectional_GRU()


def get_siamese_loss_layer(margin):
    def siamese_loss(is_different, _cosine_distance):  # 0 same ,1 different
        # the margin is only defined based on the distance function range
        return (1 - is_different) * .25 * K.pow((1 - _cosine_distance), 2) + \
               is_different * tf.where(_cosine_distance < margin, K.pow(_cosine_distance, 2), K.zeros_like(_cosine_distance))

    return siamese_loss


def vectorize_demo_data(demo_df, is_elmo):
    q1, q2, gt = demo_df[["question1"]].values[:, 0], demo_df[["question2"]].values[:, 0], demo_df[["is_duplicate"]].values[:, 0]
    q1 = tokenize_sentences_list(q1)
    q2 = tokenize_sentences_list(q2)

    if not is_elmo:
        with open("/home/elsallab/Work/cod/siamese_text/quora/quora_lang.pkl", 'rb') as f:
            _, dictionary, _ = pickle.load(f)
            q1 = vectorize_tokenized_sentences(q1, dictionary)
            q2 = vectorize_tokenized_sentences(q2, dictionary)

    return np.array(q1), np.array(q2), np.array(gt)
