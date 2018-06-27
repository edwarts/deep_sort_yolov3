from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2 as cv
import numpy as np
from PIL import Image
from yolo import YOLO
from matplotlib import pyplot as plt
import json
import argparse

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


# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort 
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

yolo = YOLO()

# lane area extremes
roi = [[200, 575], [130, 80], [260, 80], [767, 400], [767, 575], [200, 575]] # morning/AB07-1600H


track_encoder = lambda track: {"track_id": track.track_id,
                               "xywh": track.to_tlwh().tolist()}

detection_encoder = lambda det: {"xyxy": det.to_tlbr().tolist(),
                                 "class": yolo.class_names[det.cls],
                                 "score": det.score}


def video_to_json(cap_name, jfile):    
    # video capture
    vcap = cv.VideoCapture(cap_name)
    img_w = int(vcap.get(3))
    img_h = int(vcap.get(4))
    fps = vcap.get(cv.CAP_PROP_FPS)

    frame_idx = 0
    proc_fps = 0.0
    output_data = []


    while True:
        ret, frame = vcap.read()  # frame shape 640*480*3
        if ret != True:
            break;
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
        detections = [det for det in detections if yolo.class_names[det.cls] in ['person', 'bicycle', 'car',
                                                                                 'motorbike', 'aeroplane',
                                                                                 'bus', 'train', 'truck']]

        # filter detections with roi
        detections = [det for det in detections if 0 < cv.pointPolygonTest(np.array(roi),
                                                                           (det.to_xyah()[0], det.to_xyah()[1]),
                                                                           False)]

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
            bbox = track.to_tlbr()
            # update track speed in y direction, in pixels/seconds
            if not hasattr(track, 'first_frame'):
                track.first_frame = frame_idx
                track.first_bbox = track.to_tlwh()
            if frame_idx > track.first_frame + 1 * fps:
                cur_bbox = track.to_tlwh()
                cur_c_y = (cur_bbox[1] + cur_bbox[3] /2)
                first_c_y = (track.first_bbox[1] + track.first_bbox[3] /2)
                track.speed = (cur_c_y - first_c_y) / (frame_idx - track.first_frame) * fps


        # Add aggregated data
        speeds = [abs(track.speed) for track in tracker.tracks if hasattr(track, 'speed')]
        avg_spd = np.sum(speeds) / len(speeds) * 0.0005787 * 3600 # km/hr
        lane_area = cv.contourArea(np.array(roi))
        box_area = np.sum([det.tlwh[2] * det.tlwh[3] for det in detections])
        occupancy = box_area / lane_area

        proc_fps  = ( proc_fps + (1./(time.time()-t1)) ) / 2
        #print("fps= %f"%(proc_fps))

        output_data.append({
            "data_event_name": "vehicle_detection",
            "cam_id": cap_name,
            "lane_id": 1,
            "frame_idx": frame_idx,
            "average_speed": avg_spd,
            "lane_occupancy": occupancy,
            "detections": [detection_encoder(det) for det in detections],
            "tracks": [track_encoder(trk) for trk in tracker.tracks]
        })

        frame_idx += 1
        if frame_idx >= 5:
            break

    vcap.release()

    with open(jfile, 'w') as jout:
        json.dump(output_data, jout)
        

def _main(args):
    cap_name = args.cap_name
    jfile = args.jfile
    print(cap_name)
    print(jfile)
    video_to_json(cap_name, jfile)
    
    
if __name__ == "__main__":
    _main(parser.parse_args())
    
    