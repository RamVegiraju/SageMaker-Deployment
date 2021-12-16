import os
import json
import sys
import logging
import torch
from transformers import AutoTokenizer
from abc import ABC
from torch import nn
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer, BertModel
import pickle
import time
import gzip

JSON_CONTENT_TYPE = 'application/json'
PRE_TRAINED_MODEL_NAME = 'roberta-base'
CLASS_NAMES = ['negative', 'neutral', 'positive']

logger = logging.getLogger(__name__)

import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
class ModelHandler(object):
    
    def __init__(self):
        start = time.time()
        self.initialized = False
        print(f" perf __init__ {(time.time() - start) * 1000} ms")

    def initialize(self, ctx):
        start = time.time()
        #TODO
        #self.context = ctx
        #metrics = self.context.metrics        
        #stop_time = time.time()
        #metrics.add_time('initialize', round((stop_time - start_time) * 1000, 2), None, 'ms')
        
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        self.device = 'cpu'
        model_dir = properties.get('model_dir')
        
        #print('model_dir ' + model_dir)
        
        self.classes = ['not paraphrase', 'paraphrase']
        self.model = RobertaModel.from_pretrained(model_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
        self.initialized = True
        print(f" perf initialize {(time.time() - start) * 1000} ms")

    def preprocess(self, input_data):
        """
        Tokenization pre-processing
        """
        
        start = time.time()
 
        MAX_LEN = 160
        input_ids = []
        attention_masks = []
        token_type_ids = []
                
        jsontest = json.loads(input_data[0]['body'].decode())
        review_text = jsontest['text']
        encoded_review = self.tokenizer.encode_plus(
                                        review_text,
                                        max_length=MAX_LEN,
                                        add_special_tokens=True,
                                        return_token_type_ids=False,
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_tensors='pt', truncation=True
                                        )
        print(f" perf preprocess tokenizer.encode_plus {(time.time() - start) * 1000} ms")
        start = time.time()
        input_ids = encoded_review['input_ids']
        attention_mask = encoded_review['attention_mask']
        output = self.model(input_ids)
        #print(output)
        print(f" perf preprocess self.model {(time.time() - start) * 1000} ms")
        return output

    def inference(self, inputs):
        """
        Predict the class of a text using a trained transformer model.
        """
        start = time.time()
        #print(inputs)
        predictions = pickle.dumps(inputs) 
        print(f" perf inference {(time.time() - start) * 1000} ms")
        return predictions
        #return inputs

    def postprocess(self, inference_output):
        start = time.time()
        #return inference_output
        #compressed = gzip.compress(inference_output)
        print(f" postprocess {(time.time() - start) * 1000} ms")
        return [inference_output]
    
    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        start = time.time()
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        print(f" perf handle_in {(time.time() - start) * 1000} ms")
        return self.postprocess(model_output)
    
    
_service = ModelHandler()

def handle(data, context):
    start = time.time()
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None
    
    print(f" perf handle_out {(time.time() - start) * 1000} ms")
    return _service.handle(data, context)
    