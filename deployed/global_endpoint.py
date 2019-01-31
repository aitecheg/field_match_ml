"""
  Created by mohammed-alaa
"""
import json
import os
import string
import subprocess
import sys

subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'opencv-python'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pillow'])
#subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'tensorflow'])
#subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'tensorflow-hub'])
subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'prettytable'])
subprocess.call([sys.executable, '-m', 'pip', 'install', 'sagemaker==1.13.0'])

from collections import defaultdict

import boto3
import numpy as np
import prettytable
import sagemaker
import scipy.cluster.hierarchy as hcluster
from sagemaker import get_execution_role
from sagemaker.mxnet import MXNetPredictor
from sagemaker.predictor import RealTimePredictor, json_serializer, json_deserializer

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


def model_fn(model_dir):

    return None


def transform_fn(loaded_model, data, input_content_type, output_content_type):
    print('Global EP')
    
    initial_matching = json.loads(data)
    original_match = prettytable.PrettyTable(["field", "values", "field score", "value score"])
    fields = []
    values = []
    text_to_score={}
    for pair in initial_matching['field_match_output']:
        fields.append({"string": pair['field_name'], "bbox": pair['bbox'], "center": get_center(pair['bbox'])})

        text_to_score[pair['field_name']]= pair["confidence"]
        if pair["value"]['bbox'] != {'top': -1, 'height': -1, 'width': -1, 'left': -1}:
            values.append({"string": pair["value"]['field_value'], "bbox": pair["value"]['bbox'], "center": get_center(pair["value"]['bbox'])})
            text_to_score[pair["value"]['field_value']] = pair["value"]['confidence']

        # print({"strings": {"field": , "value": pair["value"]['field_value']},
        #        "bboxs": {"field": pair['bbox'], "value": pair["value"]['bbox']}})
        original_match.add_row([pair['field_name'], pair["confidence"],
                                pair["value"]['field_value'], pair["value"]['confidence']
                                ])

    print('Calling ML fields_match')
    ml_field_matching = MXNetPredictor("field-match-ml-2019-01-20")
    '''
    fields_strings = list(map(lambda item: item["string"], fields))
    values_strings = list(map(lambda item: item["string"], values))

    print(len(fields_strings))
    print(len(values_strings))
    data = {'field_names': fields_strings, 'field_values':values_strings}
    
    
    results = ml_field_matching.predict(data)
    for result in results:
        print(result)
    '''
        
    predictions_act = prettytable.PrettyTable(["field", "field score", "values", "value score", "score"])
    dist_thresh = 100
    matched_results = []
    for field in fields:
        print(field["string"])
        candidates = []
        for value in values:
            print(value["string"])
            l2_dist = l2_distance(field, value)
            if(l2_dist < dist_thresh):
                candidates.append((value, l2_dist))
                print(str(l2_dist))

        nearest = list(map(lambda item: item[0]["string"], sorted(candidates, key=lambda item: item[1])[:5]))
        input_to_matching = {"field_names": [field["string"]], "field_values": nearest}
        if(len(nearest) != 0):
            results = ml_field_matching.predict(input_to_matching)  # siamese string field match
        else:
            results = [{"field": field["string"], "value": '', "score": 0}]
            text_to_score[''] = ''
        for result in sorted(results, key=lambda item: -item["score"]):
            predictions_act.add_row([result["field"],
                                     text_to_score[result["field"]],
                                     result["value"],
                                     text_to_score[result["value"]],
                                     result["score"],
                                     ])
            matched_results.append({"field": result["field"], "value": result["value"], "score": result["score"], "field_detection_score": text_to_score[result["field"]], "value_detection_score": text_to_score[result["value"]] })

    print(predictions_act)
    
    
    #return json.dumps(results), output_content_type
    return json.dumps(matched_results), output_content_type


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
