"""
4:1 negative to positive as mentioned in http://www.aclweb.org/anthology/W16-1617
"""
import json
import os
from pprint import pprint

import numpy as np
import pandas as pd

epochs = 1
################################################################################
# root_dir = "./jsons"
################################################################################
root_dir = "jsons-triples"

################################################################################


files = os.listdir(root_dir)[:1]
print(files)
dataset = []
for epoch in range(epochs):
    page_ranges = [0]
    pairs = []
    for json_file in files:

        annotations = json.load(open(os.path.join(root_dir, json_file)))
        ################################################################################
        # for annotation in annotations:
        #     annotation = annotation[list(annotation.keys())[0]]
        #     pairs.append((annotation["field_name"].get("name"), annotation["field_value"].get("value")))
        ################################################################################
        annotation_triple = {}
        for annotation in annotations:
            if "label" in annotation:
                if annotation["label"] in annotation_triple:
                    pairs.append((annotation_triple.get("field_name"), annotation_triple.get("field_value"), 1))

                annotation_triple[annotation["label"]] = annotation["text"]
        ################################################################################
        page_ranges.append(len(pairs))

    # print(page_ranges)
    print("positive samples", len(pairs))

    for i in range(len(page_ranges) - 1):
        range_start = page_ranges[i]
        range_end = page_ranges[i + 1]

        num_negatives = min(1, range_end - range_start)
        for false_index_q1 in range(range_start, range_end):
            falses_q2 = list(np.random.randint(range_start, range_end, (num_negatives,)))
            while (false_index_q1 in falses_q2) and len(np.unique(falses_q2)) != num_negatives:
                falses_q2 = list(np.random.randint(range_start, range_end, (num_negatives,)))

            # false samples
            for false_index_q2 in falses_q2:
                pairs.append((pairs[false_index_q1][0], pairs[false_index_q2][1], 0))

            # false None
            if pairs[false_index_q1][1] is not None:  # value part is not none...make false none example to balance nones
                pairs.append((pairs[false_index_q1][0], None, 0))

    pairs = list(map(lambda item: (item[0], *item[1]), enumerate(set(pairs))))
    # pairs = list(map(lambda item: (item[0], *item[1]), enumerate(pairs)))
    print("all samples", len(pairs))
    dataset.extend(pairs)

pprint(dataset)
id_, q1, q2, label = list(zip(*dataset))
################################################################################
# df = pd.DataFrame({"id": id_,
#                    'question1': q1,
#                    'question2': q2,
#                    "is_duplicate": label
#                    })
# print("writing training dataframe", df.shape)
# df.to_csv("/home/mohammed-alaa/Downloads/train.csv", index=False)
################################################################################
df = pd.DataFrame({'test_id': id_,
                   'question1': q1,
                   "question2": q2,
                   "is_duplicate": label
                   })
print("writing testing dataframe", df.shape)
df.to_csv("/home/mohammed-alaa/Downloads/test.csv", index=False)
################################################################################
