"""
0 same ,1 different
%matplotlib inline

!wget -O all.zip "https://storage.googleapis.com/kaggle-competitions-data/kaggle/6277/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1545313629&Signature=KC0%2FXR9KOn7BrLUMyMdGsCXm%2FkvhdJJj2WvB7SOhKgh4jH8wfMm0ZfJZgJjKE1O6D0ZnI3V0qMBFo%2FVuxZl7TEsDDWpXpm00H5dYfc%2BIUyTMtdW74GlhujZcxHHqp6eaoIXrsdYEBbrrHChO4%2FOg1jT92qhgBNR8TYfaWjNjROt%2FKuUnXvjwa4KtKHH9TuqEozNPVOT3XSZq79MsW9BkmGflEtOaVI1smXN%2BmU398Rb6yqMaU%2B327kpZXMrEtIBIqmUkT1nsKvomGATpS9bxKDggnSO25sN7DDtpmApPWJh2%2BCcZEjzXn7ngknIjPHjDl9CEzYXJrxuOgR4AxeWLcQ%3D%3D"
!unzip all.zip

!wget "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec"
-------------------------
!pip install spacy -U
!pip install gensim -U
!python -m spacy download en_core_web_sm

"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
from gensim.models import FastText

from util import QuoraSequence, vectorize_tokenized_sentence, tokenize_sentences_list, sentence_demo

params_parser = argparse.ArgumentParser(description='my datareader')
params_parser.add_argument('is_quora', type=int)
params = params_parser.parse_args()

print("Is quora", params.is_quora)

is_quora = params.is_quora == 1

if is_quora:
    train_file = "/home/elsallab/Work/cod/siamese_text/quora/data/train.csv"
    test_file = "/home/elsallab/Work/cod/siamese_text/quora/data/test.csv"
else:
    train_file = "/home/elsallab/Work/cod/siamese_text/repo/data/train.csv"
    test_file = None


def read_dataframes(train_path=train_file, test_path=test_file):
    """read dataset and remove none rows"""
    train_data = pd.read_csv(train_path)

    vocab_set = set()
    remove_list = []
    for i in range(train_data.shape[0]):
        row = train_data.iloc[i]
        if type(row["question1"]) != str or type(row["question2"]) != str:
            remove_list.append(i)

    print(train_data.shape)
    print(len(remove_list))
    train_data.drop(train_data.index[remove_list], inplace=True)

    if is_quora:
        remove_list = []
        test_data = pd.read_csv(test_path)
        for i in range(test_data.shape[0]):
            row = test_data.iloc[i]
            if type(row["question1"]) != str or type(row["question2"]) != str:
                remove_list.append(i)

        print(test_data.shape)
        print(len(remove_list))

        test_data.drop(test_data.index[remove_list], inplace=True)

    return list(zip(tokenize_sentences_list(train_data["question1"], vocab_set), tokenize_sentences_list(train_data["question2"], vocab_set), list(train_data["is_duplicate"]))), \
           (list(zip(list(test_data["test_id"]), tokenize_sentences_list(test_data["question1"], vocab_set), tokenize_sentences_list(test_data["question2"], vocab_set)))) if is_quora else None, \
           vocab_set


def get_corpus(train_tuples, test_tuples):
    """make the corpus"""
    corpus_words = []
    for question1, question2, _ in train_tuples:
        corpus_words.append(question1)
        corpus_words.append(question2)

    if test_tuples is not None:
        for _, question1, question2 in test_tuples:
            corpus_words.append(question1)
            corpus_words.append(question2)

    return corpus_words


def vectorize_quora(train_tuples, test_tuples, dictionary):
    """from words to indexes"""
    transformed_train_tuples = []
    for question1, question2, is_dup in train_tuples:
        transformed_train_tuples.append(
            (vectorize_tokenized_sentence(question1, dictionary), vectorize_tokenized_sentence(question2, dictionary), is_dup)
        )

    transformed_test_tuples = None
    if test_tuples is not None:
        transformed_test_tuples = []
        for index, question1, question2 in test_tuples:
            transformed_test_tuples.append(
                (index, vectorize_tokenized_sentence(question1, dictionary), vectorize_tokenized_sentence(question2, dictionary))
            )

    return transformed_train_tuples, transformed_test_tuples


if __name__ == '__main__':
    start = time.time()
    train_tuples, test_tuples, vocab_set = read_dataframes()
    print("read & tokenized all data", time.time() - start)

    exit()
    print("textual example")
    train_sequence = QuoraSequence(train_tuples.copy(), 2)
    sentence_demo((train_sequence[0][0][0], train_sequence[0][1][0], train_sequence[0][2][0], train_sequence[0][3][0], train_sequence[0][4][0]))
    sentence_demo((train_sequence[0][0][1], train_sequence[0][1][1], train_sequence[0][2][1], train_sequence[0][3][1], train_sequence[0][4][1]))
    sentence_demo((train_sequence[1][0][0], train_sequence[1][1][0], train_sequence[1][2][0], train_sequence[1][3][0], train_sequence[1][4][0]))

    if test_tuples is not None:
        test_sequence = QuoraSequence(test_tuples.copy(), 2, is_training=False)
        sentence_demo((test_sequence[0][0][0], test_sequence[0][1][0], test_sequence[0][2][0], test_sequence[0][3][0], test_sequence[0][4][0]), is_test=True)
        sentence_demo((test_sequence[0][0][1], test_sequence[0][1][1], test_sequence[0][2][1], test_sequence[0][3][1], test_sequence[0][4][1]), is_test=True)
        sentence_demo((test_sequence[1][0][0], test_sequence[1][1][0], test_sequence[1][2][0], test_sequence[1][3][0], test_sequence[1][4][0]), is_test=True)
    ####################################################################################################################################
    train_sequence = QuoraSequence(train_tuples.copy(), 32)
    print(train_sequence[0][0].shape, train_sequence[0][1].shape, train_sequence[0][2].shape, train_sequence[0][3].shape, train_sequence[0][4].shape)

    if test_tuples is not None:
        test_sequence = QuoraSequence(test_tuples.copy(), 32, is_training=False)
        print(test_sequence[0][0], test_sequence[0][1].shape, test_sequence[0][2].shape, test_sequence[0][3].shape, test_sequence[0][4].shape)

    with open("/home/elsallab/Work/cod/siamese_text/{}/textual_data.pkl".format("quora" if is_quora else "repo"), 'wb') as f:
        pickle.dump((train_tuples, test_tuples), f)
    print("written", "/home/elsallab/Work/cod/siamese_text/{}/textual_data.pkl".format("quora" if is_quora else "repo"), ",train:", len(train_tuples),
          ",test:", len(test_tuples) if is_quora else ""
          )

    if is_quora:
        print("=" * 50)
        fasttext_model = FastText(get_corpus(train_tuples, test_tuples), size=50, window=4, min_count=2, iter=0)
        print("created initial fasttext")

        modified_vocab_set = set(fasttext_model.wv.vocab.keys())
        print("Original vocabulary", len(vocab_set))
        print("Modified vocabulary", len(modified_vocab_set))

        reverse_dictionary, dictionary = {i + 2: v for i, v in enumerate(modified_vocab_set)}, {v: i + 2 for i, v in enumerate(modified_vocab_set)}
        dictionary["<pad>"] = 0
        dictionary["<unk>"] = 1

        reverse_dictionary[0] = "<pad>"
        reverse_dictionary[1] = "<unk>"

        embeddings = np.random.rand(len(dictionary), 300).astype(np.float32) * 2 - 1

        pretrained_fasttext_model = FastText.load_fasttext_format('/home/elsallab/Work/cod/siamese_text/wiki.en')  # https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
        for word_id in reverse_dictionary:
            if reverse_dictionary[word_id] in pretrained_fasttext_model:
                embeddings[word_id] = pretrained_fasttext_model[reverse_dictionary[word_id]]
            else:
                print("not in pre-trained fasttext", reverse_dictionary[word_id])
        print("loaded embeddings")

        train_tuples_vectorized, test_tuples_vectorized = vectorize_quora(train_tuples, test_tuples, dictionary)
        print("everything vectorized")

        print("=" * 50)

        print("devectorize example")
        train_sequence = QuoraSequence(train_tuples_vectorized.copy(), 2)
        test_sequence = QuoraSequence(test_tuples_vectorized.copy(), 2, is_training=False)

        # print(train_sequence[0])
        sentence_demo((train_sequence[0][0][0], train_sequence[0][1][0], train_sequence[0][2][0], train_sequence[0][3][0], train_sequence[0][4][0]), reverse_dictionary)
        sentence_demo((train_sequence[0][0][1], train_sequence[0][1][1], train_sequence[0][2][1], train_sequence[0][3][1], train_sequence[0][4][1]), reverse_dictionary)
        sentence_demo((train_sequence[1][0][0], train_sequence[1][1][0], train_sequence[1][2][0], train_sequence[1][3][0], train_sequence[1][4][0]), reverse_dictionary)
        sentence_demo((test_sequence[0][0][0], test_sequence[0][1][0], test_sequence[0][2][0], test_sequence[0][3][0], test_sequence[0][4][0]), reverse_dictionary, is_test=True)
        sentence_demo((test_sequence[0][0][1], test_sequence[0][1][1], test_sequence[0][2][1], test_sequence[0][3][1], test_sequence[0][4][1]), reverse_dictionary, is_test=True)
        sentence_demo((test_sequence[1][0][0], test_sequence[1][1][0], test_sequence[1][2][0], test_sequence[1][3][0], test_sequence[1][4][0]), reverse_dictionary, is_test=True)

        train_sequence = QuoraSequence(train_tuples_vectorized.copy(), 32)
        test_sequence = QuoraSequence(test_tuples_vectorized.copy(), 32, is_training=False)
        print(train_sequence[0][0].shape, train_sequence[0][1].shape, train_sequence[0][2].shape, train_sequence[0][3].shape, train_sequence[0][4].shape)
        print(test_sequence[0][0], test_sequence[0][1].shape, test_sequence[0][2].shape, test_sequence[0][3].shape, test_sequence[0][4].shape)

        with open("/home/elsallab/Work/cod/siamese_text/quora/quora_lang.pkl", 'wb') as f:
            pickle.dump((embeddings, dictionary, reverse_dictionary), f)
        print("written", "/home/elsallab/Work/cod/siamese_text/quora/quora_lang.pkl")

        with open("/home/elsallab/Work/cod/siamese_text/quora/vectorized_data.pkl", 'wb') as f:
            pickle.dump((train_tuples_vectorized, test_tuples_vectorized), f)
        print("written", "/home/elsallab/Work/cod/siamese_text/quora/vectorized_data.pkl")
