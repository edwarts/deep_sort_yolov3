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

# local import
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')


cam_id = 'AB16'
start_time = 0

# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort 
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename,batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# detection model
yolo = YOLO()

# camera lane geographical info
cam_lane_info = {
    'AB03': {'roi': [[0, 365], [0, 575], [767, 575], [767, 498], [438, 170], [241, 170], [0, 365]],
             'sec_length': 4.4 * 3,
             'sections': [575, 352, 268, 225, 170],
             'speed_frames': 5},
    'AB16': {'roi': [[192, 719], [629, 500], [939, 500], [821, 719], [192, 719]],
             'sec_length': 5,
             'sections': [719, 638, 578, 522, 489],
             'speed_frames': 5}
}

# json encoder for track
track_encoder = lambda track: {"vechicle_id": track.track_id,
                               "classs": int(track.detection.cls),
                               "conf": track.detection.score,
                               #"speed": track.speed if hasattr(track, 'speed') else 0,
                               "xmin": track.to_tlwh().tolist()[0],
                               "ymin": track.to_tlwh().tolist()[1],
                               "xmax": track.to_tlwh().tolist()[0] + track.to_tlwh().tolist()[2],
                               "ymax": track.to_tlwh().tolist()[1] + track.to_tlwh().tolist()[3]}

def _init_video_handlers(cap_name, vout_name):
    vcap = cv.VideoCapture(cap_name)
    img_w = int(vcap.get(3))
    img_h = int(vcap.get(4))
    fps = vcap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    vout = cv.VideoWriter(vout_name, fourcc, fps, (img_w, img_h))
    return vcap, vout, img_w, img_h, fps

# add cv drawing to frame
def _add_cv_drawing(frame, img_w, img_h, detections, tracker, avg_spd, occupancy):
    for track in tracker.tracks:
        if track.is_confirmed() and track.time_since_update > 1 :
            continue
        bbox = track.to_tlbr()
        cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)
        cv.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])-3),0, 5e-3 * 100, (0,255,0), 1)
        
    for det in detections:
        bbox = det.to_tlbr()
        cv.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 1)
        cv.putText(frame, 
                   ' ' + yolo.class_names[det.cls] + ' ' + str(det.score),
                   (int(bbox[0])+3, int(bbox[1])-3), 0, 5e-3 * 100, (0,255,0),1)
    
    cv.putText(frame, 'detections: %d' % len(detections), (5, img_h-48), 0, 5e-3 * 150, (0,255,0), 1)
    cv.putText(frame, 'avergate speed: %.2f' % avg_spd, (5, img_h-28), 0, 5e-3 * 150, (0,255,0), 1)
    cv.putText(frame, 'occupancy: %.2f' % occupancy, (5, img_h-8), 0, 5e-3 * 150, (0,255,0), 1)

# calc real world distance between 2 points
def _calc_dist_travelled(ends, sections, sec_length):
    y1, y2 = ends
    dist = 0
    for idx, s in enumerate(sections[:-1]):
        if idx == 0:
            continue

        if y1 >= s:
            dist += (y1 - s) / (sections[idx - 1] - s) * sec_length
            #print('y1: {} {} {} {}'.format(y1, s, (y1 - s) / (sections[idx - 1] - s) * sec_length, dist))
            y1 = s
            
        if s >= y2 >= sections[idx + 1]:
            dist += (y1 - y2) / (s - sections[idx + 1]) * sec_length
            #print('y2: {} {} {} {}'.format(y2, s, (y1 - y2) / (s - sections[idx + 1]) * sec_length, dist))
            break
            
    return abs(dist)

# calc average track speed in km/hr
def _average_track_speed(tracker, fps, frame_idx, sections, sec_length, speed_frames):
    for track in tracker.tracks:
        if track.is_confirmed() and track.time_since_update > 1 :
            if hasattr(track, 'speed'):
                delattr(track, 'speed')
            continue
            
        # update track speed in y direction, in pixels/seconds
        if not hasattr(track, 'hist_frames'):
            track.hist_frames = []
            track.hist_bboxes = []
        track.hist_frames.append(frame_idx)
        track.hist_bboxes.append(track.to_tlwh())

        if len(track.hist_frames) > speed_frames:
            cur_bbox = track.to_tlwh()
            cur_c_y = (cur_bbox[1] + cur_bbox[3] /2)
            first_bbox = track.hist_bboxes[-speed_frames]
            first_c_y = first_bbox[1] + first_bbox[3] /2
            track.ends = (first_c_y, cur_c_y)

            dist_travelled = _calc_dist_travelled(track.ends, sections, sec_length)
            life = frame_idx - track.hist_frames[-speed_frames]
            speed = dist_travelled / life * fps * 3.6 # km/hr
            
            track.speed = speed
            #pdb.set_trace()
    
    track_speeds = [track.speed for track in tracker.tracks if hasattr(track, 'speed')]
    avg_spd = np.sum(track_speeds) / len(track_speeds)
    
    return avg_spd

# calc LOS
def _calc_LOS(occupancy, avg_spd):
    if occupancy > 0.7 and avg_spd < 10:
        return '4'
    elif occupancy > 0.5 and avg_spd < 30:
        return '3B'
    elif occupancy < 0.5 and avg_spd < 30:
        return '3A'
    elif occupancy > 0.5 and avg_spd < 50:
        return '2B'
    elif occupancy < 0.5 and avg_spd < 50:
        return '2A'
    elif occupancy > 0.5 and avg_spd > 50:
        return '1'
    else:
        return '0'    

# prepare detections for tracking
def extract_detections(frame, boxs, roi):
    # separate out score, class info from loc
    scores_classes = [box[-2:] for box in boxs]
    boxs = [box[0:4] for box in boxs]
    features = encoder(frame, boxs)

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
    
    return detections

# read video and output analysis in JSON
def video_to_json(cap_name, jfile, cam_id, start_time, vout_name='output.mp4', samples_per_sec=5, is_outputing_video=False):
    vcap, vout, img_w, img_h, fps = _init_video_handlers(cap_name, vout_name)

    roi = cam_lane_info[cam_id]['roi']
    sec_length = cam_lane_info[cam_id]['sec_length']
    sections = cam_lane_info[cam_id]['sections']
    speed_frames = cam_lane_info[cam_id]['speed_frames']
    
    frame_idx = 0
    proc_fps = 0.0
    output_data = []

    while True:
        ret, frame = vcap.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image(image) # x,y,w,h, score, class

        detections = extract_detections(frame, boxs, roi)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        # Add aggregated data
        avg_spd = _average_track_speed(tracker, fps, frame_idx, sections, sec_length, speed_frames)
        lane_area = cv.contourArea(np.array(roi))
        box_area = np.sum([det.tlwh[2] * det.tlwh[3] for det in detections])
        occupancy = box_area / lane_area
        los = _calc_LOS(occupancy, avg_spd)

        if is_outputing_video:
            _add_cv_drawing(frame, img_w, img_h, detections, tracker, avg_spd, occupancy)
            vout.write(frame)

        frame_idx += 1

        proc_fps  = ( proc_fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(proc_fps))

        print('{}: tracking {} vehicles, with average speed at {:.1f} km/h. {:.2f}% of the road surface is occupied.'.format(frame_idx, len(tracker.tracks), avg_spd, occupancy*100))  
        for t in tracker.tracks:
            if hasattr(t, 'speed'):
                print('{}: {} {}'.format(t.track_id, t.speed, t.ends))

        if frame_idx % fps in np.linspace(0, fps, samples_per_sec).astype(int)[:-1]:
            output_data.append({
                "data_event_name": "vehicle_detection",
                "camera_id": cam_id,
                "lane_id": 1,
                "frame_idx": frame_idx,
                "average_speed": avg_spd,
                "lane_occupancy": occupancy,
                "LOS": los,
                "time_stamp": float(start_time) + frame_idx / fps,
                "vehicles": [track_encoder(trk) for trk in tracker.tracks]
            })

        if frame_idx >= 25:
            break

# usage: python demo4.py './CAG/morning 0830/AB16-0830H (1).avi' 'AB16' 0 'output.json' 'output.mp4'
if __name__ == "__main__":
    # cap_name, jfile, vout_name, cam_id, start_time
    video_to_json(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], 5, True)