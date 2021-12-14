# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
import json
import torch 
import numpy as np
import torchvision.transforms as transforms
from six import BytesIO
import io
from PIL import Image
import cv2
import tempfile



logger = logging.getLogger(__name__)


def model_fn(model_dir):
    device = get_device()
    print('device is')
    print(device)
    model = torch.load(model_dir + '/model.pth', map_location=torch.device(device))
    print(type(model))
    return model


def input_fn(request_body, request_content_type):
    frame_width = 1024
    frame_height = 1024
    interval = 30
    f = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    print(tfile.name)
    video_frames = video2frame(tfile,frame_width, frame_height, interval)  
    #convert to tensor of float32 type
    transform = transforms.Compose([
        transforms.Lambda(lambda video_frames: torch.stack([transforms.ToTensor()(frame) for frame in video_frames])) # returns        a 4D tensor
    ])
    image_tensors = transform(video_frames)

    return image_tensors

def predict_fn(data, model):
    print('in custom predict function')
    with torch.no_grad():
        device = get_device()
        model = model.to(device)
        input_data = data.to(device)
        model.eval()
        output = model(input_data)
        
    return output

    
def output_fn(output_batch, accept='application/json'):
    res = []
    print('output list length')
    print(len(output_batch))
    for output in output_batch:
         res.append({'boxes':output['boxes'].detach().cpu().numpy().tolist(),'labels':output['labels'].detach().cpu().numpy().tolist(),'scores':output['scores'].detach().cpu().numpy().tolist()})
    
    return json.dumps(res)

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def video2frame(
    tfile,frame_width, frame_height, interval):
    """
    Extract frame from video by interval
    :param video_src_path: video src path
    :param video:　video file name
    :param frame_width:　frame width
    :param frame_height:　frame height
    :param interval:　interval for frame to extract
    :return:　list of numpy.ndarray 
    """
    video_frames = []
    cap = cv2.VideoCapture(tfile.name)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("Read failed!")

    while success:
        success, frame = cap.read()

        if frame_index % interval == 0:
            print("---> Reading the %d frame:" % frame_index, success)
            resize_frame = cv2.resize(
                frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA
            )
            video_frames.append(resize_frame)
            frame_count += 1

        frame_index += 1

    cap.release()
    print('Number of frames')
    print(frame_count)
    return video_frames
