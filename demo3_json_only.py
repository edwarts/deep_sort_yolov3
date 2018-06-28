from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2 as cv
import numpy as np
from PIL import Image
from yolo import YOLO
import json
import argparse
import math

# local import
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Video to json')
parser.add_argument('cap_name', help='Path to input video file.')
parser.add_argument('jfile', help='Path to JSON output file.')
parser.add_argument('cam_id', help='Camera id.')
parser.add_argument('start_time', help='Start timestamp.')


# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort 
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

yolo = YOLO()

track_encoder = lambda track: {"vechicle_id": track.track_id,
                               "classs": int(track.detection.cls),
                               "conf": track.detection.score,
                               "xmin": track.to_tlwh().tolist()[0],
                               "ymin": track.to_tlwh().tolist()[1],
                               "xmax": track.to_tlwh().tolist()[0] + track.to_tlwh().tolist()[2],
                               "ymax": track.to_tlwh().tolist()[1] + track.to_tlwh().tolist()[3]}

def video_to_json(cap_name, jfile, cam_id, start_time):
    # video capture
    vcap = cv.VideoCapture(cap_name)
    fps = int(math.ceil(vcap.get(cv.CAP_PROP_FPS)))

    frame_idx = 0
    proc_fps = 0.0
    output_data = []
    timestamp = int(start_time)

    while True:
        ret, frame = vcap.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image) # x,y,w,h, score, class

        # separate out score, class info from loc
        scores_classes = [box[-2:] for box in boxs]
        boxs = [box[0:4] for box in boxs]
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # filter unwanted obj by classfication
        for idx, det in list(enumerate(detections)):
            det.score = "%.2f" % scores_classes[idx][0]
            det.cls = scores_classes[idx][1]
        detections = [det for det in detections if yolo.class_names[det.cls] in ['person', 
                                                                                 'bicycle', 
                                                                                 'car',
                                                                                 'motorbike',
                                                                                 'aeroplane',
                                                                                 'bus', 
                                                                                 'train', 
                                                                                 'truck']]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update > 1 :
                continue

        proc_fps  = ( proc_fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %f"%(proc_fps))

        if frame_idx % fps == 0:
            output_data.append({
                "data_event_name": "vehicle_detection",
                "camera_id": cam_id,
                "time_stamp": timestamp,
                "vehicles": [track_encoder(trk) for trk in tracker.tracks]
            })
            timestamp += 1

        frame_idx += 1
        if frame_idx > 30: break

    vcap.release()

    with open(jfile, 'w') as jout:
        json.dump(output_data, jout)
        

def _main(args):
    cap_name = args.cap_name
    jfile = args.jfile
    cam_id = args.cam_id
    start_time = args.start_time
    video_to_json(cap_name, jfile, cam_id, start_time)
    
    
if __name__ == "__main__":
    _main(parser.parse_args())
    
    