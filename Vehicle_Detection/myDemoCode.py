# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 01:32:13 2018

@author: Sunny
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


import cv2
from matplotlib import pyplot as plt
from PIL import Image


path = 'G:\\VDAS\\Cropped Videos'
output_path = 'G:\\VDAS\\Masked Videos'

videoList = os.listdir(path) 
numberOfVideos = len(videoList)
print(numberOfVideos)
i=2

while(i<numberOfVideos):
    
    input_path = path+'\\'+videoList[i]
    output_path_video=output_path+"\\"+videoList[i]+"masked.avi"
    #failed_frames_path = "C:\\Users\\Sunny\\Desktop\\VDAS\\Unstitched\\Video_"+str(i)
    directory = output_path
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    verbose_flag = False
    total_count = 0
    count_stitched = 0
    count_lost = 0    
    videoCapture = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    frame_size = (int(videoCapture.get(3)/3),
                    int(videoCapture.get(4)))
    size = (int(videoCapture.get(3)),
            int(videoCapture.get(4)))
    videoWriter = cv2.VideoWriter(output_path_video, 
                                          fourcc,
                                          10, 
                                          size)
    
    success, frame = videoCapture.read()
    frame_index = 0
    
    alpha = 0.4
    beta = 1 - alpha
    while(success):        
            # Run detection
            results = model.detect([frame], verbose=1)

            # Visualize results
            r = results[0]

            new_image = visualize.display_instances(frame, 
                                        r['rois'], 
                                        r['masks'], 
                                        r['class_ids'],
                                        class_names, 
                                        r['scores'])
            #new_image = cv2.resize(new_image,frame_size,interpolation=cv2.INTER_CUBIC)
            #print("before make image:",type(new_image))
            #new_image = new_image.make_image(None)
            #print("after make image:",type(new_image))

            videoWriter.write(new_image)
            success, frame = videoCapture.read()
    i+=1
videoCapture.release()
videoWriter.release()
print("Detection Finished")

