
import json
import os
import string
import subprocess
import sys

subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'opencv-python'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pillow'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'tensorflow'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'tensorflow-hub'])
subprocess.call([sys.executable, '-m', 'pip', 'install', 'sagemaker==1.13.0'])

from collections import defaultdict

import boto3
import sagemaker
import scipy.cluster.hierarchy as hcluster
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNetPredictor
from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer
import traceback

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

from tensorflow.python.keras.engine.saving import model_from_json

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub
from tensorflow.python.layers.base import Layer
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell

role = get_execution_role()
session = boto3.Session(region_name='us-west-2')
sagemaker_session = sagemaker.Session(boto_session=session)


def l2_distance(field, value):
    return np.linalg.norm((np.array(field["center"]) - np.array(value["center"])))


def get_center(bbox):  # {'top': 911, 'height': 31, 'width': 328, 'left': 961}
    return bbox['top'] + bbox['height'] / 2, bbox["left"] + bbox["width"] / 2


class JSONPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)


class CustomElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        self.elmo = None
        self.result = None
        super(CustomElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False,
                               )

        self._trainable_weights += self.elmo._graph._collections["variables"]  # building multiple times will accumulate the weights
        print("added ", len(self._trainable_weights), "variables")
        super(CustomElmoEmbeddingLayer, self).build(input_shape)

    def call(self, tokenized, mask=None):
        tokens_input, tokens_length = tokenized

        self.result = self.elmo(inputs={
            "tokens": tokens_input,
            "sequence_len": tokens_length[:, 0]
        },
            as_dict=True,
            signature='tokens',
        )['elmo']  # [batch_size, max_length, 1024]
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


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


# !/usr/bin/python
"""
Implementation of the Hungarian (Munkres) Algorithm using Python and NumPy
References: http://www.ams.jhu.edu/~castello/362/Handouts/hungarian.pdf
        http://weber.ucsd.edu/~vcrawfor/hungar.pdf
        http://en.wikipedia.org/wiki/Hungarian_algorithm
        http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
        http://www.clapper.org/software/python/munkres/
"""

# Module Information.
__version__ = "1.1.1"
__author__ = "Thom Dedecko"
__url__ = "http://github.com/tdedecko/hungarian-algorithm"
__copyright__ = "(c) 2010 Thom Dedecko"
__license__ = "MIT License"


class HungarianError(Exception):
    pass


# Import numpy. Error if fails
try:
    import numpy as np
except ImportError:
    raise HungarianError("NumPy is not installed.")


class Hungarian:
    """
    Implementation of the Hungarian (Munkres) Algorithm using np.
    Usage:
        hungarian = Hungarian(cost_matrix)
        hungarian.calculate()
    or
        hungarian = Hungarian()
        hungarian.calculate(cost_matrix)
    Handle Profit matrix:
        hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
    or
        cost_matrix = Hungarian.make_cost_matrix(profit_matrix)
    The matrix will be automatically padded if it is not square.
    For that numpy's resize function is used, which automatically adds 0's to any row/column that is added
    Get results and total potential after calculation:
        hungarian.get_results()
        hungarian.get_total_potential()
    """

    def __init__(self, input_matrix=None, is_profit_matrix=False):
        """
        input_matrix is a List of Lists.
        input_matrix is assumed to be a cost matrix unless is_profit_matrix is True.
        """
        if input_matrix is not None:
            # Save input
            my_matrix = np.array(input_matrix)
            self._input_matrix = np.array(input_matrix)
            self._maxColumn = my_matrix.shape[1]
            self._maxRow = my_matrix.shape[0]

            # Adds 0s if any columns/rows are added. Otherwise stays unaltered
            matrix_size = max(self._maxColumn, self._maxRow)
            pad_columns = matrix_size - self._maxRow
            pad_rows = matrix_size - self._maxColumn
            my_matrix = np.pad(my_matrix, ((0, pad_columns), (0, pad_rows)), 'constant', constant_values=(0))

            # Convert matrix to profit matrix if necessary
            if is_profit_matrix:
                my_matrix = self.make_cost_matrix(my_matrix)

            self._cost_matrix = my_matrix
            self._size = len(my_matrix)
            self._shape = my_matrix.shape

            # Results from algorithm.
            self._results = []
            self._totalPotential = 0
        else:
            self._cost_matrix = None

    def get_results(self):
        """Get results after calculation."""
        return self._results

    def get_total_potential(self):
        """Returns expected value after calculation."""
        return self._totalPotential

    def calculate(self, input_matrix=None, is_profit_matrix=False):
        """
        Implementation of the Hungarian (Munkres) Algorithm.
        input_matrix is a List of Lists.
        input_matrix is assumed to be a cost matrix unless is_profit_matrix is True.
        """
        # Handle invalid and new matrix inputs.
        if input_matrix is None and self._cost_matrix is None:
            raise HungarianError("Invalid input")
        elif input_matrix is not None:
            self.__init__(input_matrix, is_profit_matrix)

        result_matrix = self._cost_matrix.copy()

        # Step 1: Subtract row mins from each row.
        for index, row in enumerate(result_matrix):
            result_matrix[index] -= row.min()

        # Step 2: Subtract column mins from each column.
        for index, column in enumerate(result_matrix.T):
            result_matrix[:, index] -= column.min()

        # Step 3: Use minimum number of lines to cover all zeros in the matrix.
        # If the total covered rows+columns is not equal to the matrix size then adjust matrix and repeat.
        total_covered = 0
        while total_covered < self._size:
            # Find minimum number of lines to cover all zeros in the matrix and find total covered rows and columns.
            cover_zeros = CoverZeros(result_matrix)
            covered_rows = cover_zeros.get_covered_rows()
            covered_columns = cover_zeros.get_covered_columns()
            total_covered = len(covered_rows) + len(covered_columns)

            # if the total covered rows+columns is not equal to the matrix size then adjust it by min uncovered num (m).
            if total_covered < self._size:
                result_matrix = self._adjust_matrix_by_min_uncovered_num(result_matrix, covered_rows, covered_columns)

        # Step 4: Starting with the top row, work your way downwards as you make assignments.
        # Find single zeros in rows or columns.
        # Add them to final result and remove them and their associated row/column from the matrix.
        expected_results = min(self._maxColumn, self._maxRow)
        zero_locations = (result_matrix == 0)
        while len(self._results) != expected_results:

            # If number of zeros in the matrix is zero before finding all the results then an error has occurred.
            if not zero_locations.any():
                raise HungarianError("Unable to find results. Algorithm has failed.")

            # Find results and mark rows and columns for deletion
            matched_rows, matched_columns = self.__find_matches(zero_locations)

            # Make arbitrary selection
            total_matched = len(matched_rows) + len(matched_columns)
            if total_matched == 0:
                matched_rows, matched_columns = self.select_arbitrary_match(zero_locations)

            # Delete rows and columns
            for row in matched_rows:
                zero_locations[row] = False
            for column in matched_columns:
                zero_locations[:, column] = False

            # Save Results
            self.__set_results(zip(matched_rows, matched_columns))

        # Calculate total potential
        value = 0
        for row, column in self._results:
            value += self._input_matrix[row, column]
        self._totalPotential = value

    @staticmethod
    def make_cost_matrix(profit_matrix):
        """
        Converts a profit matrix into a cost matrix.
        Expects NumPy objects as input.
        """
        # subtract profit matrix from a matrix made of the max value of the profit matrix
        matrix_shape = profit_matrix.shape
        offset_matrix = np.ones(matrix_shape, dtype=int) * profit_matrix.max()
        cost_matrix = offset_matrix - profit_matrix
        return cost_matrix

    def _adjust_matrix_by_min_uncovered_num(self, result_matrix, covered_rows, covered_columns):
        """Subtract m from every uncovered number and add m to every element covered with two lines."""
        # Calculate minimum uncovered number (m)
        elements = []
        for row_index, row in enumerate(result_matrix):
            if row_index not in covered_rows:
                for index, element in enumerate(row):
                    if index not in covered_columns:
                        elements.append(element)
        min_uncovered_num = min(elements)

        # Add m to every covered element
        adjusted_matrix = result_matrix
        for row in covered_rows:
            adjusted_matrix[row] += min_uncovered_num
        for column in covered_columns:
            adjusted_matrix[:, column] += min_uncovered_num

        # Subtract m from every element
        m_matrix = np.ones(self._shape, dtype=int) * min_uncovered_num
        adjusted_matrix -= m_matrix

        return adjusted_matrix

    def __find_matches(self, zero_locations):
        """Returns rows and columns with matches in them."""
        marked_rows = np.array([], dtype=int)
        marked_columns = np.array([], dtype=int)

        # Mark rows and columns with matches
        # Iterate over rows
        for index, row in enumerate(zero_locations):
            row_index = np.array([index])
            if np.sum(row) == 1:
                column_index, = np.where(row)
                marked_rows, marked_columns = self.__mark_rows_and_columns(marked_rows, marked_columns, row_index,
                                                                           column_index)

        # Iterate over columns
        for index, column in enumerate(zero_locations.T):
            column_index = np.array([index])
            if np.sum(column) == 1:
                row_index, = np.where(column)
                marked_rows, marked_columns = self.__mark_rows_and_columns(marked_rows, marked_columns, row_index,
                                                                           column_index)

        return marked_rows, marked_columns

    @staticmethod
    def __mark_rows_and_columns(marked_rows, marked_columns, row_index, column_index):
        """Check if column or row is marked. If not marked then mark it."""
        new_marked_rows = marked_rows
        new_marked_columns = marked_columns
        if not (marked_rows == row_index).any() and not (marked_columns == column_index).any():
            new_marked_rows = np.insert(marked_rows, len(marked_rows), row_index)
            new_marked_columns = np.insert(marked_columns, len(marked_columns), column_index)
        return new_marked_rows, new_marked_columns

    @staticmethod
    def select_arbitrary_match(zero_locations):
        """Selects row column combination with minimum number of zeros in it."""
        # Count number of zeros in row and column combinations
        rows, columns = np.where(zero_locations)
        zero_count = []
        for index, row in enumerate(rows):
            total_zeros = np.sum(zero_locations[row]) + np.sum(zero_locations[:, columns[index]])
            zero_count.append(total_zeros)

        # Get the row column combination with the minimum number of zeros.
        indices = zero_count.index(min(zero_count))
        row = np.array([rows[indices]])
        column = np.array([columns[indices]])

        return row, column

    def __set_results(self, result_lists):
        """Set results during calculation."""
        # Check if results values are out of bound from input matrix (because of matrix being padded).
        # Add results to results list.
        for result in result_lists:
            row, column = result
            if row < self._maxRow and column < self._maxColumn:
                new_result = (int(row), int(column))
                self._results.append(new_result)


class CoverZeros:
    """
    Use minimum number of lines to cover all zeros in the matrix.
    Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
    """

    def __init__(self, matrix):
        """
        Input a matrix and save it as a boolean matrix to designate zero locations.
        Run calculation procedure to generate results.
        """
        # Find zeros in matrix
        self._zero_locations = (matrix == 0)
        self._shape = matrix.shape

        # Choices starts without any choices made.
        self._choices = np.zeros(self._shape, dtype=bool)

        self._marked_rows = []
        self._marked_columns = []

        # marks rows and columns
        self.__calculate()

        # Draw lines through all unmarked rows and all marked columns.
        self._covered_rows = list(set(range(self._shape[0])) - set(self._marked_rows))
        self._covered_columns = self._marked_columns

    def get_covered_rows(self):
        """Return list of covered rows."""
        return self._covered_rows

    def get_covered_columns(self):
        """Return list of covered columns."""
        return self._covered_columns

    def __calculate(self):
        """
        Calculates minimum number of lines necessary to cover all zeros in a matrix.
        Algorithm based on: http://weber.ucsd.edu/~vcrawfor/hungar.pdf
        """
        while True:
            # Erase all marks.
            self._marked_rows = []
            self._marked_columns = []

            # Mark all rows in which no choice has been made.
            for index, row in enumerate(self._choices):
                if not row.any():
                    self._marked_rows.append(index)

            # If no marked rows then finish.
            if not self._marked_rows:
                return True

            # Mark all columns not already marked which have zeros in marked rows.
            num_marked_columns = self.__mark_new_columns_with_zeros_in_marked_rows()

            # If no new marked columns then finish.
            if num_marked_columns == 0:
                return True

            # While there is some choice in every marked column.
            while self.__choice_in_all_marked_columns():
                # Some Choice in every marked column.

                # Mark all rows not already marked which have choices in marked columns.
                num_marked_rows = self.__mark_new_rows_with_choices_in_marked_columns()

                # If no new marks then Finish.
                if num_marked_rows == 0:
                    return True

                # Mark all columns not already marked which have zeros in marked rows.
                num_marked_columns = self.__mark_new_columns_with_zeros_in_marked_rows()

                # If no new marked columns then finish.
                if num_marked_columns == 0:
                    return True

            # No choice in one or more marked columns.
            # Find a marked column that does not have a choice.
            choice_column_index = self.__find_marked_column_without_choice()

            while choice_column_index is not None:
                # Find a zero in the column indexed that does not have a row with a choice.
                choice_row_index = self.__find_row_without_choice(choice_column_index)

                # Check if an available row was found.
                new_choice_column_index = None
                if choice_row_index is None:
                    # Find a good row to accomodate swap. Find its column pair.
                    choice_row_index, new_choice_column_index = \
                        self.__find_best_choice_row_and_new_column(choice_column_index)

                    # Delete old choice.
                    self._choices[choice_row_index, new_choice_column_index] = False

                # Set zero to choice.
                self._choices[choice_row_index, choice_column_index] = True

                # Loop again if choice is added to a row with a choice already in it.
                choice_column_index = new_choice_column_index

    def __mark_new_columns_with_zeros_in_marked_rows(self):
        """Mark all columns not already marked which have zeros in marked rows."""
        num_marked_columns = 0
        for index, column in enumerate(self._zero_locations.T):
            if index not in self._marked_columns:
                if column.any():
                    row_indices, = np.where(column)
                    zeros_in_marked_rows = (set(self._marked_rows) & set(row_indices)) != set([])
                    if zeros_in_marked_rows:
                        self._marked_columns.append(index)
                        num_marked_columns += 1
        return num_marked_columns

    def __mark_new_rows_with_choices_in_marked_columns(self):
        """Mark all rows not already marked which have choices in marked columns."""
        num_marked_rows = 0
        for index, row in enumerate(self._choices):
            if index not in self._marked_rows:
                if row.any():
                    column_index, = np.where(row)
                    if column_index in self._marked_columns:
                        self._marked_rows.append(index)
                        num_marked_rows += 1
        return num_marked_rows

    def __choice_in_all_marked_columns(self):
        """Return Boolean True if there is a choice in all marked columns. Returns boolean False otherwise."""
        for column_index in self._marked_columns:
            if not self._choices[:, column_index].any():
                return False
        return True

    def __find_marked_column_without_choice(self):
        """Find a marked column that does not have a choice."""
        for column_index in self._marked_columns:
            if not self._choices[:, column_index].any():
                return column_index

        raise HungarianError(
            "Could not find a column without a choice. Failed to cover matrix zeros. Algorithm has failed.")

    def __find_row_without_choice(self, choice_column_index):
        """Find a row without a choice in it for the column indexed. If a row does not exist then return None."""
        row_indices, = np.where(self._zero_locations[:, choice_column_index])
        for row_index in row_indices:
            if not self._choices[row_index].any():
                return row_index

        # All rows have choices. Return None.
        return None

    def __find_best_choice_row_and_new_column(self, choice_column_index):
        """
        Find a row index to use for the choice so that the column that needs to be changed is optimal.
        Return a random row and column if unable to find an optimal selection.
        """
        row_indices, = np.where(self._zero_locations[:, choice_column_index])
        for row_index in row_indices:
            column_indices, = np.where(self._choices[row_index])
            column_index = column_indices[0]
            if self.__find_row_without_choice(column_index) is not None:
                return row_index, column_index

        # Cannot find optimal row and column. Return a random row and column.
        from random import shuffle

        shuffle(row_indices)
        column_index, = np.where(self._choices[row_indices[0]])
        return row_indices[0], column_index[0]


def tokenize_sentences_list(sentences_list, vocab_set=None):
    """Tokenize list of sentences"""

    tokenized_list = list(map(tokenize_sentence, list(sentences_list)))

    if vocab_set is not None:
        for tokenized_sentence in tokenized_list:
            for word in tokenized_sentence:
                vocab_set.add(word)
    return tokenized_list


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


PADDING_TOKEN = "þ"


def pad_sequences_wrapped(sequence_list):
    is_str = type(sequence_list[0][0]) == str
    return pad_sequences(sequence_list, padding="post", truncating="post", dtype=object if is_str else np.int32, value=PADDING_TOKEN if is_str else 0.)


def vectorize_demo_data(q1, q2):
    q1 = tokenize_sentences_list(q1)
    q2 = tokenize_sentences_list(q2)

    q1_len = list(map(lambda item: len(item), q1))
    q2_len = list(map(lambda item: len(item), q2))

    q1 = pad_sequences_wrapped(q1)
    q2 = pad_sequences_wrapped(q2)

    return np.array(q1), np.array(q1_len).reshape((-1, 1)), np.array(q2), np.array(q2_len).reshape((-1, 1))


model_root = None
######################################

state_size = 300
staked_layers = 3


######################################
def possible_pairs(vectorized_field_names, vectorized_field_values):
    field_names_side = []
    field_values_side = []
    for vectorized_field_name in vectorized_field_names:
        for vectorized_field_value in vectorized_field_values:
            field_names_side.append(vectorized_field_name)
            field_values_side.append(vectorized_field_value)
    return field_names_side, field_values_side


def model_fn(model_dir):
    print("starting :model_fn")
    global model_root
    model_root = model_dir
    with K.get_session().graph.as_default():
        print("midd :model_fn")
        siamese = model_from_json(open(os.path.join(model_root, 'w8s.arch'), "r").read(), custom_objects={'_custom_stacked_bidirectional_GRU': custom_stacked_bidirectional_GRU_layer(state_size, staked_layers), "CustomElmoEmbeddingLayer": CustomElmoEmbeddingLayer, "ManDistanceLayer": ManDistanceLayer})
        siamese.load_weights(os.path.join(model_root, "w8s.h5"))
    print("finishing :model_fn")
    return siamese


def local_ml_pairing(data, loaded_model):
    field_names = data["field_names"]
    field_values = data["field_values"]

    field_names_side, field_values_side = possible_pairs(field_names, field_values)
    field_vec, field_len, value_vec, value_len = vectorize_demo_data(field_names_side, field_values_side)

    with K.get_session().graph.as_default():
        prediction = loaded_model.predict([field_vec, field_len, value_vec, value_len])

    ####################################################
    cost_matrix = []
    for field_index in range(len(field_names)):
        predictions = prediction[field_index * len(field_values):(field_index + 1) * len(field_values)]
        cost_matrix.append((1 - predictions).squeeze(axis=-1).tolist())

    ####################################################
    # hangarian approach
    hungarian = Hungarian(cost_matrix)
    hungarian.calculate()
    # ["field":"string","value":"string"]
    new_response = []
    for h_result in hungarian.get_results():
        h_field, h_value = h_result
        new_response.append({"field": field_names[h_field], "value": field_values[h_value], "score": 1 - cost_matrix[h_field][h_value]})
    return new_response


def transform_fn(loaded_model, data, input_content_type, output_content_type):
    parsed = json.loads(data)

    loc_endpoint = parsed.get("loc_endpoint", "localization-model-2019-01-29")
    fm_endpoint = parsed.get("fm_endpoint", 'field-match-2019-01-24-12-39-05-522')

    hw_endpoint = parsed.get("hw_endpoint", "pytorch-handwriting-ocr-2019-01-29-02-06-44-538")
    hp_endpoint = parsed.get("hp_endpoint", "hand-printed-model-2019-01-29-1")
    sp_endpoint = parsed.get("sp_endpoint", "hand-printed-model-2019-01-29-1")

    # access keys
    aws_access_key_id = parsed.get("aws_access_key_id", None)
    aws_secret_access_key = parsed.get("aws_secret_access_key", None)

    bucket = parsed.get("bucket")
    file_name = parsed.get("file_name")

    loc_predictor = MXNetPredictor(loc_endpoint, sagemaker_session)
    field_matching = JSONPredictor(fm_endpoint, sagemaker_session)
    try:
        loc_out = loc_predictor.predict({"url": "s3://{}/{}".format(bucket, file_name)})
    except Exception as ex:
        print(ex)
        tb = traceback.format_exc()
        # return error here
    print("localized")
    loc_out = loc_out["result"]
    print(loc_out)

    data = {
        "hw_endpoint": hw_endpoint,
        "hp_endpoint": hp_endpoint,  # ''  #
        "sp_endpoint": sp_endpoint,

        "field_names": [{"bucket": "ahmedb-test", "filename": "field_name_list.txt"},
                        {"bucket": "unum-files", "filename": "unum_field_names.txt"}],
        "field_names_ignore": [
            {"bucket": "ahmedb-test", "filename": "must_ignore.txt"},
            {"bucket": "unum-files", "filename": "unum_must_ignore_field_names.txt"}
        ],

        "hw_pickle": {"bucket": loc_out['bucket_name'], "filename": loc_out['hw_key']},
        "hp_pickle": {"bucket": loc_out['bucket_name'], "filename": loc_out['hp_key']},
        "page_image": {"bucket": bucket, "filename": file_name},

    }

    fields = []
    values = []
    text_to_score = {}
    bbox_of_all = {}
    try:
        initial_matching = field_matching.predict(data)
    except Exception as ex:
        print(ex)
        tb = traceback.format_exc()

    for pair in initial_matching['field_match_output']:
        fields.append({"string": pair['field_name'], "bbox": pair['bbox'], "center": get_center(pair['bbox'])})
        bbox_of_all[pair['field_name']] = pair['bbox']
        text_to_score[pair['field_name']] = pair["confidence"]
        if pair["value"]['bbox'] != {'top': -1, 'height': -1, 'width': -1, 'left': -1}:
            values.append({"string": pair["value"]['field_value'], "bbox": pair["value"]['bbox'], "center": get_center(pair["value"]['bbox'])})
            text_to_score[pair["value"]['field_value']] = pair["value"]['confidence']
            bbox_of_all[pair["value"]['field_value']] = pair["value"]['bbox']

    points_2d = []
    for field in fields:
        points_2d.append(field["center"])
    for value in values:
        points_2d.append(value["center"])

    points_2d = np.array(points_2d)

    # clustering
    thresh = 250
    clusters = hcluster.fclusterdata(points_2d, thresh, criterion="distance")

    groupings = defaultdict(lambda: {'field_names': [], 'field_values': []})
    for index, class_ in enumerate(clusters):
        if index >= len(fields):
            groupings[class_]["field_values"].append({"string": values[index - len(fields)]["string"], "center": values[index - len(fields)]["center"]})
        else:
            groupings[class_]["field_names"].append({"string": fields[index]["string"], "center": fields[index]["center"]})

    for value in groupings.values():
        field_names_centers = list(map(lambda item: item["center"], value['field_names'])) + list(map(lambda item: item["center"], value['field_values']))
        if field_names_centers:
            center = np.mean(field_names_centers, axis=0)
        else:
            center = np.array([np.inf, np.inf])

        value['field_names'] = list(map(lambda item: item["string"], value['field_names']))
        value['field_values'] = list(map(lambda item: item["string"], value['field_values']))
        value['center'] = center

    while True:  # merging things
        not_ready_for_matching_list = []
        for key in groupings:
            value = groupings[key]
            if len(value["field_values"]) > len(value["field_names"]):
                not_ready_for_matching_list.append((key, value))

        if len(not_ready_for_matching_list) == 0:
            break  # enough merging

        for key, not_ready_for_matching in not_ready_for_matching_list:
            del groupings[key]
            distances = sorted(list(map(lambda item: (item, np.linalg.norm(groupings[item]["center"] - not_ready_for_matching["center"])), groupings)), key=lambda item: item[1])
            groupings[distances[1][0]] = {"field_values": groupings[distances[1][0]]["field_values"] + not_ready_for_matching["field_values"],
                                          "field_names": groupings[distances[1][0]]["field_names"] + not_ready_for_matching["field_names"],
                                          "center": np.mean([groupings[distances[1][0]]["center"], not_ready_for_matching["center"]], axis=0)
                                          }

    final_output_json = []  # list of those {'value_detection_score': '', 'value': '', 'field_detection_score': 0.9559999999999998, 'score': 0, 'field': 'ATTENDING PHYSICIAN STATEMENT '}

    for cluster in [grouping for grouping in groupings.values()]:
        cluster = {"field_names": list(set(cluster["field_names"])), "field_values": list(set(cluster["field_values"]))}
        if cluster["field_names"] and cluster["field_values"]:
            results = local_ml_pairing(cluster, loaded_model)

            for result in sorted(results, key=lambda item: -item["score"]):
                final_output_json.append({'value_detection_score': text_to_score[result["value"]],
                                          'value': result["value"],
                                          'field_detection_score': text_to_score[result["field"]],
                                          'score': result["score"],
                                          'field': result["field"],
                                          'value_bbox': bbox_of_all[result["value"]],
                                          'field_bbox': bbox_of_all[result["field"]]
                                          }

                                         )
        else:
            for field_name in cluster["field_names"]:
                final_output_json.append(
                    {'value_detection_score': 0,
                     'value': '',
                     'field_detection_score': text_to_score[field_name],
                     'score': 0,
                     'field': field_name,
                     'value_bbox': {'width': -1, 'top': -1, 'height': -1, 'left': -1},
                     'field_bbox': bbox_of_all[field_name]
                     }

                )

    return json.dumps(final_output_json), output_content_type

############

# with tf.device('/device:GPU:{}'.format(1)):
#     the_model = model_fn(".")
#     a, _ = transform_fn(the_model, json.dumps({"field_names": ["have you been hospitalized?", "ph@ysicina's tax d number:", "my sposue:", "first\" name", "physician's tax id number"],
#                                                "field_values": ["yes", "Tanya Thornton", "Catherine", "786 535 6586", "ø"]}), "fasf", "asf")
#
#     results = json.loads(a)
#
#     for result in results:
#         print(result)
