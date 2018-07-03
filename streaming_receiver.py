from __future__ import division, print_function, absolute_import
from kafka import KafkaConsumer
from datetime import datetime
import glob
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
samples_per_sec = 5  # no. of outputs per second

# lane area extremes
roi = [[200, 575], [130, 80], [260, 80], [767, 400], [767, 575], [200, 575]]  # morning/AB07-1600H
topic = 'test-video-streaming'
consumer = KafkaConsumer(topic, group_id='view', bootstrap_servers=['127.0.0.1:9092'])


def video_stream_to_json(cap_name, jfile, cam_id,start_time):
    # video capture
    frame_idx = 0
    proc_fps = 0.0
    output_data = []

    seconds = 1
    yolo_detect_avg = 0.0
    fps_avvg = 0.0
    feature_encode_avg = 0.0
    feature_encode_avg_25 = 0.0
    yolo_detect_avg_25 = 0.0
    fps_avg_25 = 0.0

    initial = time.time()
    for each_message in consumer:
        t1 = time.time()

        # img=np.asarray(bytearray(each_message.value), dtype=np.uint8)
        # image = Image.fromarray(img)
        input_data=np.fromstring(each_message.value, np.uint8)
        # input_data=Image.frombytes(mode='I',size=(416, 416),data=each_message.value)
        # input_data=cv.imdecode(each_message.value,'.jpeg')
        print(input_data)
        # image = cv.imdecode('.jpeg',input_data)
        image=Image.fromarray(input_data)
        #
        # # img = cv.imdecode(nparr, cv.CV)
        # # image = Image.fromarray(each_message.value)
        # t1_yolo_start = time.time()
        boxs = yolo.detect_image(image)  # x,y,w,h, score, class
        # t2_yolo_end = time.time()
        # yolo_detect_avg += t2_yolo_end - t1_yolo_start
        # yolo_detect_avg_25 += t2_yolo_end - t1_yolo_start
        # # separate out score, class info from loc
        scores_classes = [box[-2:] for box in boxs]
        boxs = [box[0:4] for box in boxs]
        # f_t_start = time.time()
        features = encoder(image, boxs)
        # f_t_end = time.time()
        # feature_encode_avg += f_t_end - f_t_start
        # feature_encode_avg_25 += f_t_end - f_t_start
        #
        # # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        #
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
        # filter detections with roi
        detections = [det for det in detections if 0 < cv.pointPolygonTest(np.array(roi),
                                                                           (det.to_xyah()[0],
                                                                            det.to_xyah()[1]),
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
            if track.is_confirmed() and track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # update track speed in y direction, in pixels/seconds
            if not hasattr(track, 'first_frame'):
                track.first_frame = frame_idx
                track.first_bbox = track.to_tlwh()
            if frame_idx > track.first_frame + 1 * 25:
                cur_bbox = track.to_tlwh()
                cur_c_y = (cur_bbox[1] + cur_bbox[3] / 2)
                first_c_y = (track.first_bbox[1] + track.first_bbox[3] / 2)
                track.speed = (cur_c_y - first_c_y) / (frame_idx - track.first_frame) * 25

        # Add aggregated data
        speeds = [abs(track.speed) for track in tracker.tracks if hasattr(track, 'speed')]
        avg_spd = np.sum(speeds) / len(speeds) * 0.0005787 * 3600  # km/hr
        lane_area = cv.contourArea(np.array(roi))
        box_area = np.sum([det.tlwh[2] * det.tlwh[3] for det in detections])
        occupancy = box_area / lane_area

        proc_fps = (proc_fps + (1. / (time.time() - t1))) / 2
        # print("fps= %f"%(proc_fps))
        fps_avvg += proc_fps
        fps_avg_25 += proc_fps
        output_data={
                "data_event_name": "vehicle_detection",
                "camera_id": cam_id,
                "lane_id": 1,
                "frame_idx": frame_idx,
                "average_speed": avg_spd,
                "lane_occupancy": occupancy,
                "time_stamp": float(start_time) + frame_idx / 25,
                "vehicles": [track_encoder(trk) for trk in tracker.tracks]
            }
        print(output_data)

        # if frame_idx % 25 in np.linspace(0, 25, samples_per_sec).astype(int)[:-1]:
        #     output_data.append({
        #         "data_event_name": "vehicle_detection",
        #         "camera_id": cam_id,
        #         "lane_id": 1,
        #         "frame_idx": frame_idx,
        #         "average_speed": avg_spd,
        #         "lane_occupancy": occupancy,
        #         "time_stamp": float(start_time) + frame_idx / 25,
        #         "vehicles": [track_encoder(trk) for trk in tracker.tracks]
        #     })
        #
        # frame_idx += 1
        #
        # if frame_idx % 25 == 0:
        #     print("THe {} second:".format(str(seconds)))
        #     seconds += 1
        #     print("Total frame {}".format(str(frame_idx)))
        #     print("Average Speed for FPS is {}, for Yolo computing is {} second, for feature encoding is {} second"
        #           .format(fps_avg_25 / float(25.0), yolo_detect_avg_25 / float(25.0),
        #                   feature_encode_avg_25 / float(25.0)))
        #     yolo_detect_avg_25 = 0
        #     feature_encode_avg_25 = 0
        #     fps_avg_25 = 0

        # if frame_idx > 30: break
    t_final = time.time()
    # print("Final Average Speed for FPS is {}, for Yolo computing is {} second, for feature encoding is {} second"
    #    .format(fps_avvg/frame_idx, yolo_detect_avg / frame_idx, feature_encode_avg / frame_idx))
    # print("The total time for this process is {}".format(t_final-initial))
    # with open(jfile, 'w') as jout:
    #     json.dump(output_data, jout)




def timestamp_convert(filename):
    date = filename.split("-")[1]
    time = filename.split("-")[2]
    dt = date[0:4] + '-' + date[4:6] + '-' + date[6:8] + ' ' + time[0:2] + ':' + time[2:] + ':00'
    ts = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").timestamp()
    return ts


def process_video_files_from_folder(root_folder, output_folder):
    # full_categories = [x[0] for x in os.walk(root_folder) if x[0]][1:]
    print(root_folder)
    data = []
    filenames = next(os.walk(root_folder))[2]

    print(filenames)
    for file_name in filenames:
        first_part = file_name.split(".")[0]
        cap_name = file_name
        jfile = first_part + '.json'
        cam_id = first_part.split("-")[0]
        start_time = timestamp_convert(first_part)
        print({'cap_name': cap_name, 'jfile_name': jfile, 'cam_id': cam_id,
               'timestamp': start_time})
        # video_to_json(root_folder +'/'+cap_name, output_folder +'/'+jfile, cam_id, start_time)
        data.append(
            {'cap_name': cap_name, 'jfile_name': jfile, 'cam_id': cam_id,
             'timestamp': start_time})


if __name__ == "__main__":
    video_stream_to_json(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    # Lauch kafka consumer


