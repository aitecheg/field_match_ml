import argparse
import logging
import pickle

import gensim
import numpy as np
import pandas as pd

from util import tokenize_sentence

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

params_parser = argparse.ArgumentParser(description='repo')
params_parser.add_argument('train', type=str)
params_parser.add_argument('test', type=str)
params = params_parser.parse_args()
embedding_dim = 250


def extract_questions():
    """
    Extract questions for making word2vec model.
    """
    df1 = pd.read_csv("./data/{}.csv".format(params.train), keep_default_na=False)
    df2 = pd.read_csv("./data/{}.csv".format(params.test), keep_default_na=False)

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 100000 == 0:
                print("example tokenization", row['question1'], tokenize_sentence(row['question1']))
            if i != 0 and i % 10000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['question1']:
                yield tokenize_sentence(row['question1'])
            if row['question2']:
                yield tokenize_sentence(row['question2'])


documents = list(extract_questions())
logging.info("Done reading data file")

model = gensim.models.FastText(documents, size=embedding_dim, iter=20, min_count=3)
print("vocab size", len(model.wv.vocab))
vocab_dict = {word: index + 2 for index, word in enumerate(list(model.wv.vocab))}
reverse_dict = {index: word for word, index in vocab_dict.items()}
# 0,1 are reserved
# 1 for unk
# 0 for padding

embeddings = 1 * np.random.randn(len(vocab_dict) + 2, embedding_dim)  # This will be the embedding matrix
for word, index in vocab_dict.items():
    embeddings[index] = model[word]

with open("./models/Quora-Question-Pairs.ft", 'wb') as f:
    pickle.dump((vocab_dict, reverse_dict, embeddings), f)  # dumps get the string
