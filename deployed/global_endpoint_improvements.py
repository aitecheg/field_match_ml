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

'''
Inputs:
- doc_img: tiff
- filed_bbox: {'field': ,'field_bbox':{'top': , 'height': , 'width': , 'left': }}
- candidates_bbox: [{'value': ,'value_bbox':{'top': , 'height': , 'width': , 'left': }},
                    {'value': ,'value_bbox':{'top': , 'height': , 'width': , 'left': }},
                    {'value': ,'value_bbox':{'top': , 'height': , 'width': , 'left': }},
                    ....]
Outputs:
- Draw rect on "doc_img" in "color" the field_bbox in continous, and all candidates_bbox in dashed.
'''
def visualize_candidates(doc_img, field, candidates, color):
    return

'''
Inputs:
- doc_img: tiff
- mached_pairs_bbox: [{'field': ,'field_bbox': {'top': , 'height': , 'width': , 'left': }, 'value':, 'value_bbox': {'top': , 'height': , 'width': , 'left': }},
{'field': ,'field_bbox': {'top': , 'height': , 'width': , 'left': }, 'value':, 'value_bbox': {'top': , 'height': , 'width': , 'left': }}
{'field': ,'field_bbox': {'top': , 'height': , 'width': , 'left': }, 'value':, 'value_bbox': {'top': , 'height': , 'width': , 'left': }}
...
]
                
Outputs:
Draw a line between bboxes between the pairs bboxes. on the doc_img
'''
def visualize_matches(doc_img, mached_pairs_bbox):
    return

def detect_fields(initial_matching):
    '''
    fields = []
    values = []
    bbox_of_all = {}
    text_to_score={}
    for pair in initial_matching['field_match_output']:
        fields.append({"string": pair['field_name'], "bbox": pair['bbox'], "center": get_center(pair['bbox'])})
        bbox_of_all[pair['field_name']] = pair['bbox']
        text_to_score[pair['field_name']]= pair["confidence"]
        if pair["value"]['bbox'] != {'top': -1, 'height': -1, 'width': -1, 'left': -1}:
            values.append({"string": pair["value"]['field_value'], "bbox": pair["value"]['bbox'], "center": get_center(pair["value"]['bbox'])})
            text_to_score[pair["value"]['field_value']] = pair["value"]['confidence']
            bbox_of_all[pair["value"]['field_value']] = pair["value"]['bbox']

  
    return fields, values, bbox_of_all, text_to_score
    '''

def get_candidates(fields, values):
    '''
    fields_with_candidates = fields
    for field in fields:
        #print(field["string"])
        candidates = []
        for value in values:
            #print(value["string"])
            l2_dist = l2_distance(field, value)
            if(l2_dist < dist_thresh):
                candidates.append((value, l2_dist))
                #print(str(l2_dist))

        nearest = list(map(lambda item: item[0]["string"], sorted(candidates, key=lambda item: item[1])[:5]))   
        fields_with_candidates[field["string"]]['candidates'] = nearest
        
    return fields_with_candidates
    '''

    
def get_match_score_ml(fields_with_candidates, bbox_of_all, text_to_score):    
    ml_field_matching = MXNetPredictor("field-match-ml-2019-01-20-1")    
    matched_results = []
    ''' NN
    for field in fields_with_candidates:
        
        input_to_matching = {"field_names": [field["string"]], "field_values": fields_with_candidates[field["string"]]['candidates']}
        if(len(nearest) != 0):
            results = ml_field_matching.predict(input_to_matching)  # siamese string field match
        else:
            results = [{"field": field["string"], "value": '', "score": 0}]
            text_to_score[''] = ''
            bbox_of_all[''] = {'width': -1, 'top': -1, 'height': -1, 'left': -1}
            
        for result in sorted(results, key=lambda item: -item["score"]):
            matched_results.append({"field": result["field"], 
                                    "value": result["value"], 
                                    "score": result["score"], 
                                    "field_detection_score": text_to_score[result["field"]], 
                                    "value_detection_score": text_to_score[result["value"]], 
                                    "value_bbox": bbox_of_all[result["value"]], 
                                    "field_bbox": bbox_of_all[result["field"]] })
    return results
    '''
    ''' Hundarian
    '''
     
def get_match_score_rule_based(initial_matching, field):
    '''
    for pair in initial_matching['field_match_output']:
        if pair['field_name'] == field['string']
            return {"field": pair['field_name'],"value": pair["value"]['field_value'], "score": pair['confidence'],"field_detection_score": pair['confidence'],"value_detection_score": pair["value"]['confidence'],"value_bbox": pair["value"]['bbox'],"field_bbox":pair['bbox']}#pair#pair["confidence"]
    '''
    
'''
Inputs:
- ml_pair: {"field": ,"value": , "score": ,"field_detection_score": ,"value_detection_score": ,"value_bbox": ,"field_bbox":] }
- rule_based_pair: {"field": ,"value": , "score": ,"field_detection_score": ,"value_detection_score": ,"value_bbox": ,"field_bbox":] }
Outputs:
result_pair: {"field": ,"value": , "score": ,"field_detection_score": ,"value_detection_score": ,"value_bbox": ,"field_bbox":] }
'''
def fuse_ml_and_rule_based(ml_pair, rule_based_pair):
    ''' Greedy max
    temp = [ml_pair, rule_based_pair]
    result_pair = temp[np.armax(ml_pair['score'], result_pair['score'])]
    return result_pair
    '''
def ml_field_match_pipeline(data):
    # Detect
    initial_matching = json.loads(data)
    fields, values, bbox_of_all, text_to_score = detect_fields(initial_matching)
    
    # Candidates
    fields_with_candidates = get_candidates(fields, values)
    
    # Score
    ml_pairs = get_match_score_ml(fields_with_candidates, bbox_of_all, text_to_score)
    
    
    # Match
    results = []   
    for field in fields:
        results.append(fuse_ml_and_rule_based(ml_pairs[field], get_match_score_rule_based(initial_matching, field)))
    
    return results
    
        
class JSONPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(JSONPredictor, self).__init__(endpoint_name, sagemaker_session, json_serializer, json_deserializer)


def model_fn(model_dir):

    return None

# data: {'fields_detected': # output of old fields match, 'doc': path to tiff img on s3}
def transform_fn(loaded_model, data, input_content_type, output_content_type):
    print('Global EP')
    initial_matching = json.loads(data)
    #parsed = json.loads(data)
    #initial_matching = parsed['fields_detected']
    #doc_img = parsed['doc']
    original_match = prettytable.PrettyTable(["field", "field score", "values", "value score"])
    fields = []
    values = []
    bbox_of_all = {}
    text_to_score={}
    matched_results_dict = {}
    field_id = 0
    for pair in initial_matching['field_match_output']:
        fields.append({"id": field_id, "string": pair['field_name'], "bbox": pair['bbox'], "center": get_center(pair['bbox'])})
        bbox_of_all[pair['field_name']] = pair['bbox']
        text_to_score[pair['field_name']]= pair["confidence"]
        if pair["value"]['bbox'] != {'top': -1, 'height': -1, 'width': -1, 'left': -1}:
            values.append({"string": pair["value"]['field_value'], "bbox": pair["value"]['bbox'], "center": get_center(pair["value"]['bbox'])})
            text_to_score[pair["value"]['field_value']] = pair["value"]['confidence']
            bbox_of_all[pair["value"]['field_value']] = pair["value"]['bbox']

        # print({"strings": {"field": , "value": pair["value"]['field_value']},
        #        "bboxs": {"field": pair['bbox'], "value": pair["value"]['bbox']}})
        original_match.add_row([pair['field_name'], pair["confidence"],
                                pair["value"]['field_value'], pair["value"]['confidence']
                                ])
        matched_results_dict[field_id] = {'field': pair['field_name'],
                             "value": pair["value"]['field_value'], 
                             "score": pair["confidence"], 
                             "field_detection_score": pair["confidence"], 
                             "value_detection_score": pair["value"]['confidence'], 
                             "value_bbox": pair["value"]['bbox'], 
                             "field_bbox": pair['bbox'] }
        field_id += 1

    print('Calling ML fields_match')
    ml_field_matching = MXNetPredictor("field-match-ml-2019-01-20-1")
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
    ml_matched_results = []
    ml_matched_results_dict = {}
    # n_colors = 10
    #colors = {0: 'black', 1: 'red', 2: 'blue',..., n_colors:'magenta'}
    
    for field in fields:
        #print(field["string"])
        candidates = []
        for value in values:
            #print(value["string"])
            l2_dist = l2_distance(field, value)
            if(l2_dist < dist_thresh):
                candidates.append((value, l2_dist))
                #print(str(l2_dist))

        nearest = list(map(lambda item: item[0]["string"], sorted(candidates, key=lambda item: item[1])[:5]))
        input_to_matching = {"field_names": [field["string"]], "field_values": nearest}
        #visualize_candidates(doc_img=doc_img, field=field, candidates=input_to_matching, color=colors[np.randint(0,n_colors)])
        if(len(nearest) != 0):
            results = ml_field_matching.predict(input_to_matching)  # siamese string field match
        else:
            results = [{"field": field["string"], "value": '', "score": 0}]
            text_to_score[''] = 0
            bbox_of_all[''] = {'width': -1, 'top': -1, 'height': -1, 'left': -1}
        for result in sorted(results, key=lambda item: -item["score"]):
            predictions_act.add_row([result["field"],
                                     text_to_score[result["field"]],
                                     result["value"],
                                     text_to_score[result["value"]],
                                     result["score"],
                                     ])
            ml_matched_results.append({"field": result["field"], 
                                    "value": result["value"], 
                                    "score": result["score"], 
                                    "field_detection_score": text_to_score[result["field"]], 
                                    "value_detection_score": text_to_score[result["value"]], 
                                    "value_bbox": bbox_of_all[result["value"]], 
                                    "field_bbox": bbox_of_all[result["field"]] })
            ml_matched_results_dict[field["id"]] = {"field": result["field"],
                                                "value": result["value"], 
                                                 "score": result["score"], 
                                                 "field_detection_score": text_to_score[result["field"]], 
                                                 "value_detection_score": text_to_score[result["value"]], 
                                                 "value_bbox": bbox_of_all[result["value"]], 
                                                 "field_bbox": bbox_of_all[result["field"]] }  

    # Fuse textual with visual clues
    final_matched_results = []
    for field_id in ml_matched_results_dict:
        if(matched_results_dict[field_id]['value_detection_score'] == 0): matched_results_dict[field_id]['value_detection_score'] = 1
        if(matched_results_dict[field_id]['value_detection_score'] > ml_matched_results_dict[field_id]['value_detection_score']):
            final_matched_results.append(matched_results_dict[field_id])
        else:
            final_matched_results.append(ml_matched_results_dict[field_id])
            
           
    #visualize_matches(doc_img, mached_pairs_bbox)
    #print(predictions_act)
    print('finished')
    
    #return json.dumps(results), output_content_type
    return json.dumps(final_matched_results), output_content_type


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
