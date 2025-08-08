#!/usr/bin/env python
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import logging
import numpy as np
import time
import os
import joblib
from djl_python import Input
from djl_python import Output


class SKLearnRegressor(object):
    def __init__(self):
        self.initialized = False

    def initialize(self, properties: dict):
        """
        Initialize model.
        """
        print(os.listdir())
        if os.path.exists("model.joblib"):
            self.model = joblib.load(os.path.join("model.joblib"))
        else:
            raise ValueError("Expecting a model.joblib artifact for SKLearn Model Loading")
        self.initialized = True

    def inference(self, inputs):
        """
        Custom service entry point function.

        :param inputs: the Input object holds a list of numpy array
        :return: the Output object to be send back
        """

        #sample input: [[0.5]]
        
        try:
            data = inputs.get_as_json()
            print(data)
            print(type(data))
            res = self.model.predict(data).tolist()[0]
            outputs = Output()
            outputs.add_as_json(res)
        except Exception as e:
            logging.exception("inference failed")
            # error handling
            outputs = Output().error(str(e))
        
        print(outputs)
        print(type(outputs))
        print("Returning inference---------")
        return outputs


_service = SKLearnRegressor()


def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())
    
    if inputs.is_empty():
        return None

    return _service.inference(inputs)