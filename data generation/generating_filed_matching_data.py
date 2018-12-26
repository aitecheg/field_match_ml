"""
4:1 negative to positive as mentioned in http://www.aclweb.org/anthology/W16-1617
"""
import json
import os
from random import random, sample, shuffle

import numpy as np
import pandas as pd

epochs = 25
negative_samples_count = 1
################################################################################
# root_dir = "./jsons"
################################################################################
root_dir = "jsons-triples"


# root_dir = "jsons-new"


################################################################################

def rectarea(r):
    """Return the area of rectangle <r>"""
    return (r[1][0] - r[0][0]) * (r[1][1] - r[0][1])


def e_distance(bbox1, bbox2):
    # np.array([[left, top], [right, bottom]]
    c1 = (bbox1[0] + bbox1[1]) / 2
    c2 = (bbox2[0] + bbox2[1]) / 2
    return np.linalg.norm(c2 - c1)


def rectintersect(a, b):
    """ returns index 0 ~ 1
    normalizes with respect to the second"""
    a_left, a_top, a_right, a_bottom = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
    b_left, b_top, b_right, b_bottom = b[0, 0], b[0, 1], b[1, 0], b[1, 1]

    if a_right <= b_left or b_right <= a_left or b_bottom <= a_top or a_bottom <= b_top:
        return 0

    intersection_box = np.array([[max(a[0, 0], b[0, 0]), max(a[0, 1], b[0, 1])], [min(a[1, 0], b[1, 0]), min(a[1, 1], b[1, 1])]])

    area = rectarea(intersection_box)

    return area / rectarea(b)


def search_boxes(field_area_bbox, list_of_bboxes, accept_none=False):
    found = []
    for bbox in list_of_bboxes:
        if rectintersect(field_area_bbox, bbox["bbox"]) > .85:
            found.append(bbox)

    if not found and accept_none:
        found = [{"text": None}]
    return found


def unique(my_list):
    class hashabledict(dict):
        def __hash__(self):
            return hash(tuple(sorted(self.items())))

    return set(map(lambda item: hashabledict(item), my_list))


files = os.listdir(root_dir)

test_file_id, sanity_file_id = sample(range(len(files)), 2)
test_file, sanity_file = files[test_file_id], files[sanity_file_id]

new_format = False
for annotation in json.load(open(os.path.join(root_dir, "1.json"))):
    field_area_key = list(annotation.keys())[0]
    if "field_area" in field_area_key and field_area_key != "field_area":
        new_format = True

print("Expected new format:", new_format)
print("=" * 50)
# test_file = "2.json"
# test_file_id = 12

train_dataset = []
test_dataset = []
demo_dataset = []
sanity_demo_dataset = []
stat_field_area = {}
for epoch in range(epochs):
    page_ranges = [0]
    pairs = []
    for json_file in files:

        is_testing = (test_file == json_file)
        is_sanity = (sanity_file == json_file)

        annotations = json.load(open(os.path.join(root_dir, json_file)))
        # # ######################################################################################
        # # #                                New format Generator                                #
        # # ######################################################################################
        #
        value_bboxes = []
        for annotation in annotations:
            field_area_key = list(annotation.keys())[0]
            if "field_area" in field_area_key:
                annotation = annotation[field_area_key]

                value = annotation["field_value"]
                if value == {}:
                    continue
                left, right, top, bottom = value["bbox"]["x"], value["bbox"]["x"] + value["bbox"]["l"], value["bbox"]["y"], value["bbox"]["y"] + value["bbox"]["w"]
                if left - right == 0 or top - bottom == 0:
                    continue

                value_bboxes.append({"bbox": np.array([[left, top], [right, bottom]]), "text": value["value"]})

        for annotation in annotations:
            field_area_key = list(annotation.keys())[0]
            if "field_area" in field_area_key:
                annotation = annotation[field_area_key]
                pairs.append((annotation["field_name"].get("name"), annotation["field_value"].get("value"), 1))

                value = annotation["field_value"]
                if value == {}:
                    continue
                left, right, top, bottom = value["bbox"]["x"], value["bbox"]["x"] + value["bbox"]["l"], value["bbox"]["y"], value["bbox"]["y"] + value["bbox"]["w"]
                if left - right == 0 or top - bottom == 0:
                    continue

                if is_testing and epoch == 0:
                    demo_dataset.append((annotation["field_name"].get("name"), annotation["field_value"].get("value"), 1))
                    distances = sorted(list(map(lambda item: (e_distance(np.array([[left, top], [right, bottom]]), item["bbox"]), item), value_bboxes)), key=lambda item: item[0])
                    for neighbour in distances[:5]:
                        if annotation["field_value"].get("value") != neighbour[1]["text"]:
                            demo_dataset.append((annotation["field_name"].get("name"), neighbour[1]["text"], 0))

                if is_sanity and epoch == 0:
                    sanity_demo_dataset.append((annotation["field_name"].get("name"), annotation["field_value"].get("value"), 1))
                    distances = sorted(list(map(lambda item: (e_distance(np.array([[left, top], [right, bottom]]), item["bbox"]), item), value_bboxes)), key=lambda item: item[0])
                    for neighbour in distances[:5]:
                        if annotation["field_value"].get("value") != neighbour[1]["text"]:
                            sanity_demo_dataset.append((annotation["field_name"].get("name"), neighbour[1]["text"], 0))

        stat_field_area[json_file] = len(annotations)
        # ######################################################################################
        #                                  old format Generator                                #
        # ######################################################################################
        name_bboxes = []
        value_bboxes = []
        set_of_field_area = set()
        # search candidates
        for i, annotation in enumerate(annotations):
            if "label" in annotation:
                if annotation["label"] == "field_name":
                    left, right, top, bottom = annotation["left"], annotation["left"] + annotation["width"], annotation["top"], annotation["top"] + annotation["height"]
                    if left - right == 0 or top - bottom == 0:
                        continue
                    name_bboxes.append({"bbox": np.array([[left, top], [right, bottom]]), "text": annotation["text"]})

                if annotation["label"] == "field_value":
                    left, right, top, bottom = annotation["left"], annotation["left"] + annotation["width"], annotation["top"], annotation["top"] + annotation["height"]
                    if left - right == 0 or top - bottom == 0:
                        continue
                    value_bboxes.append({"bbox": np.array([[left, top], [right, bottom]]), "text": annotation["text"]})

                if annotation["label"] == "field_area":
                    set_of_field_area.add((annotation["left"], annotation["top"], annotation["width"], annotation["height"]))

        stat_field_area[json_file] = len(set_of_field_area)
        # making pairs
        for annotation in annotations:
            if "label" in annotation and annotation["label"] == "field_area":
                left, right, top, bottom = annotation["left"], annotation["left"] + annotation["width"], annotation["top"], annotation["top"] + annotation["height"]
                if left - right == 0 or top - bottom == 0:
                    continue

                area_bbox = np.array([[left, top], [right, bottom]])

                name_inside = search_boxes(area_bbox, name_bboxes)
                values_inside = search_boxes(area_bbox, value_bboxes, accept_none=True)

                # if len(name_inside) > 1:
                #     print(name_inside, values_inside, "\n")
                # if len(values_inside) > 1 and len(name_inside) > 1:
                #     print(name_inside, "\n", values_inside, "\n")
                # if len(values_inside) > 1:
                #     print(name_inside, values_inside, "\n")

                for name in name_inside:
                    for value in values_inside:
                        # getting negative neighbours for demo data
                        if is_testing and epoch == 0:
                            demo_dataset.append((name["text"], value["text"], 1))
                            distances = sorted(list(map(lambda item: (e_distance(name["bbox"], item["bbox"]), item), value_bboxes)), key=lambda item: item[0])
                            for neighbour in distances[:5]:
                                if neighbour[1]["text"] != value["text"]:
                                    demo_dataset.append((name["text"], neighbour[1]["text"], 0))

                        # getting negative neighbours for sanity data
                        if is_sanity and epoch == 0:
                            sanity_demo_dataset.append((name["text"], value["text"], 1))
                            distances = sorted(list(map(lambda item: (e_distance(name["bbox"], item["bbox"]), item), value_bboxes)), key=lambda item: item[0])
                            for neighbour in distances[:5]:
                                if neighbour[1]["text"] != value["text"]:
                                    sanity_demo_dataset.append((name["text"], neighbour[1]["text"], 0))

                        pairs.append((name["text"], value["text"], 1))

        ################################################################################
        page_ranges.append(len(pairs))

    # print(page_ranges)
    print("positive samples", len(pairs))
    negative_testing_pairs = []
    for i in range(len(page_ranges) - 1):
        range_start = page_ranges[i]
        range_end = page_ranges[i + 1]

        num_negatives = min(negative_samples_count, range_end - range_start)
        for false_index_q1 in range(range_start, range_end):
            falses_q2 = list(np.random.randint(range_start, range_end, (num_negatives,)))
            while (false_index_q1 in falses_q2) and len(np.unique(falses_q2)) != num_negatives:
                falses_q2 = list(np.random.randint(range_start, range_end, (num_negatives,)))

            ######################################################################
            # false samples
            for false_index_q2 in falses_q2:
                if i == test_file_id:
                    negative_testing_pairs.append((pairs[false_index_q1][0], pairs[false_index_q2][1], 0))
                else:
                    pairs.append((pairs[false_index_q1][0], pairs[false_index_q2][1], 0))

            # false None samples
            if random() > .5 and pairs[false_index_q1][1] is not None:  # value part is not none...make false none example to balance nones
                if i == test_file_id:
                    negative_testing_pairs.append((pairs[false_index_q1][0], None, 0))
                else:
                    pairs.append((pairs[false_index_q1][0], None, 0))
            ######################################################################

    pairs = list(map(lambda item: (item[0], *item[1]), enumerate(list(pairs))))
    testing_pairs = list(map(lambda item: (item[0], *item[1]), enumerate(list(negative_testing_pairs))))

    print("all samples", len(pairs) + len(testing_pairs))
    test_dataset.extend(testing_pairs + pairs[page_ranges[test_file_id]:page_ranges[test_file_id + 1]])
    train_dataset.extend(pairs[:page_ranges[test_file_id]] + pairs[page_ranges[test_file_id + 1]:])

# pprint(dataset)

sanity_dataset = list(map(lambda item: (item[0], *item[1]), enumerate(list(sanity_demo_dataset))))
############################################################################
shuffle(train_dataset)
id_, q1, q2, label = list(zip(*(train_dataset + sanity_dataset)))
print("unique field names", len(set(q1)))
print("unique field values", len(set(q2)))
train_df = pd.DataFrame({"id": id_,
                         'question1': q1,
                         'question2': q2,
                         "is_duplicate": label
                         })
print("positive labels ratio", np.mean(train_df["is_duplicate"]))

print("=" * 100)
print("train_df:positive labels ratio", np.mean(train_df["is_duplicate"]))
print("writing training dataframe", train_df.shape)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_df.head(n=10))
    print(train_df.tail(n=10))
train_df.to_csv("/home/mohammed-alaa/Downloads/train.csv", index=False)
############################################################################
print("=" * 100)
shuffle(test_dataset)
id_, q1, q2, label = list(zip(*test_dataset))
test_df = pd.DataFrame({"id": id_,
                        'question1': q1,
                        'question2': q2,
                        "is_duplicate": label
                        })
print("positive labels ratio", np.mean(test_df["is_duplicate"]))

print("writing testing dataframe", test_df.shape)
print("test_df:positive labels ratio", np.mean(test_df["is_duplicate"]))
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(test_df.head(n=10))
    print(test_df.tail(n=10))
test_df.to_csv("/home/mohammed-alaa/Downloads/test.csv", index=False)
############################################################################
print("=" * 100)
demo_dataset = list(map(lambda item: (item[0], *item[1]), enumerate(list(demo_dataset))))
id_, q1, q2, label = list(zip(*demo_dataset))
demo_df = pd.DataFrame({"id": id_,
                        'question1': q1,
                        'question2': q2,
                        "is_duplicate": label
                        })
print("positive labels ratio", np.mean(demo_df["is_duplicate"]))

print("demo_df:positive labels ratio", np.mean(demo_df["is_duplicate"]))
print("writing demo dataframe", demo_df.shape)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(demo_df)
demo_df.to_csv("/home/mohammed-alaa/Downloads/demo.csv", index=False)
############################################################################
print("=" * 100)

id_, q1, q2, label = list(zip(*sanity_dataset))
sanity_df = pd.DataFrame({"id": id_,
                          'question1': q1,
                          'question2': q2,
                          "is_duplicate": label
                          })
print("positive labels ratio", np.mean(sanity_df["is_duplicate"]))

print("sanity_df:positive labels ratio", np.mean(sanity_df["is_duplicate"]))
print("writing demo dataframe", sanity_df.shape)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(sanity_df)
sanity_df.to_csv("/home/mohammed-alaa/Downloads/sanity.csv", index=False)
############################################################################
print("=" * 100)
print("statistics")
print(stat_field_area)
