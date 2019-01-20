"""
  Created by mohammed-alaa
"""
import json

import cv2
import matplotlib.pyplot as plt

################ PARAMETERS ####################
page = 2
show_every_annotation = True
################################################
json_file = "Accident Claim - 1/Page {}-.json".format(page)
image_file = "Accident Claim - 1/Page {}.jpg".format(page)

image = cv2.imread(image_file)
annotations = json.load(open(json_file))

for annotation in annotations:
    annotation = annotation[list(annotation.keys())[0]]
    pair_bbox = annotation["bbox"]  # {'bbox':{'x': 163, 'y': 559, 'l': 1096, 'w': 75}}

    field_name_bbox = annotation["field_name"].get("bbox", None)  # {'bbox': {'x': 163, 'y': 559, 'l': 599, 'w': 34}, 'name': 'Please check the type of claim you are filing:'}
    if field_name_bbox is not None:
        cv2.rectangle(image, (field_name_bbox["x"], field_name_bbox["y"]), (field_name_bbox["x"] + field_name_bbox["l"], field_name_bbox["y"] + field_name_bbox["w"]), (0, 255, 0), 8)

    field_value_bbox = annotation["field_value"].get("bbox", None)  # {'bbox': {'x': 163, 'y': 559, 'l': 599, 'w': 34}, 'value': 'Accidental Injury'}
    if field_value_bbox is not None:
        cv2.rectangle(image, (field_value_bbox["x"], field_value_bbox["y"]), (field_value_bbox["x"] + field_value_bbox["l"], field_value_bbox["y"] + field_value_bbox["w"]), (0, 0, 255), 2)

    if show_every_annotation:
        print("field_name", annotation["field_name"])
        print("field_value", annotation["field_value"])
        cv2.imshow("annotation", image[pair_bbox["y"]:pair_bbox["y"] + pair_bbox["w"], pair_bbox["x"]:pair_bbox["x"] + pair_bbox["l"]])
        cv2.waitKey()

    cv2.rectangle(image, (pair_bbox["x"], pair_bbox["y"]), (pair_bbox["x"] + pair_bbox["l"], pair_bbox["y"] + pair_bbox["w"]), (255, 0, 0), 7)

plt.imshow(image)
plt.show()
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow("image", image)
# cv2.waitKey()
