import argparse
import logging

import gensim
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

params_parser = argparse.ArgumentParser(description='repo')
params_parser.add_argument('train', type=str)
params_parser.add_argument('test', type=str)
params = params_parser.parse_args()


def extract_questions():
    """
    Extract questions for making word2vec model.
    """
    df1 = pd.read_csv("./data/{}.csv".format(params.train), keep_default_na=False)
    df2 = pd.read_csv("./data/{}.csv".format(params.test), keep_default_na=False)

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            if i != 0 and i % 1000 == 0:
                logging.info("read {0} sentences".format(i))

            if row['question1']:
                yield gensim.utils.simple_preprocess(row['question1'])
            if row['question2']:
                yield gensim.utils.simple_preprocess(row['question2'])


documents = list(extract_questions())
logging.info("Done reading data file")

model = gensim.models.FastText(documents, size=300, iter=20)
# model.train(documents, total_examples=len(documents), epochs=10)
model.save("./models/Quora-Question-Pairs.ft")
