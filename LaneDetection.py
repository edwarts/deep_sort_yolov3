
# coding: utf-8

# In[8]:


from __future__ import division, print_function, absolute_import

import os
import math
from matplotlib import pyplot as plt

import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


class Tracking:
    def __init__(self):
        self.id = None
        self.centroids = []
        self.trend = []
        self.bboxes = []
        self.lane = None
        self.slope = None
        self.intersection = None
        self.frames = []
        
    def __init__(self, id, trend, centroids, bboxes, frames):
        self.id = id
        self.centroids = centroids
        self.trend = trend
        self.bboxes = bboxes
        self.slope = self.trend[0]
        self.intersection = self.trend[0]
        self.lane = None
        self.frames = frames
        
    def associate_to_closest_lane(self, lanes):
        min_degrees = 361
        for lane in lanes:
            diff = abs(slope_to_degree(self.slope) - slope_to_degree(lane.slope))
            if diff < min_degrees:
                min_degrees = diff
                self.lane = lane
    
    def calc_bbox_change_ratio(self):
        area = lambda coords : (coords[2] - coords[0]) * (coords[3] - coords[1])
        return area(self.bboxes[0]) / area(self.bboxes[1])
    
    def calc_bbox_speed(self):
        length = lambda centroids : math.sqrt((centroids[1][0] - centroids[0][0])**2 +                                               (centroids[1][1] - centroids[0][1])**2)
        return length([self.centroids[0], self.centroids[-1]]) / len(self.frames)


# In[10]:


class Lane:
    def __init__(self):
        self.trend = None
        self.end_pts = []
        self.history = 0
        self.slope = None
        self.intersection = None
        self.average_bbox_change_ratio = 1
        self.average_bbox_speed = 0
        
    def __init__(self, trend, end_pts):
        self.trend = trend
        self.end_pts = end_pts
        self.history = 0
        self.slope = self.trend[0]
        self.intersection = self.trend[0]
        self.average_bbox_change_ratio = 1
        self.average_bbox_speed = 0
        
    def merge(self, trend, trend_end_pts):
        self.trend = (self.trend + trend) / 2
        trend_len = (trend_end_pts[1][0] - trend_end_pts[0][0])**2 + (trend_end_pts[1][1] - trend_end_pts[0][1])**2
        lane_len = (self.end_pts[1][0] - self.end_pts[0][0])**2 + (self.end_pts[1][1] - self.end_pts[0][1])**2
        if (trend_len > lane_len):
            self.end_pts[0][0] = (self.end_pts[0][0] + trend_end_pts[0][0]) / 2
            self.end_pts[0][1] = (self.end_pts[0][1] + trend_end_pts[0][1]) / 2
            self.end_pts[1][0] = (self.end_pts[1][0] + trend_end_pts[1][0]) / 2
            self.end_pts[1][1] = (self.end_pts[1][1] + trend_end_pts[1][1]) / 2
    
    def calc_average_bbox_change_ratio(self, trackings):
        lane_trackings = [tracking for tracking in trackings if tracking.lane is lane]
        ratio_sum = sum([tracking.calc_bbox_change_ratio() for tracking in lane_trackings])
        self.average_bbox_change_ratio = ratio_sum / len(lane_trackings)
    
    def calc_average_bbox_speed(self, trackings):
        lane_trackings = [tracking for tracking in trackings if tracking.lane is lane]  
        speed_sum = sum([tracking.calc_bbox_speed() for tracking in lane_trackings])
        self.average_bbox_speed = speed_sum / len(lane_trackings)


# In[11]:


def read_detection_file(detection_file_path):
    detection_file = open(detection_file_path, 'r')
    detection_lines = detection_file.readlines()
    detection_file.close()
    
    detections = []
    detections_frame = {}

    for line in detection_lines:
        line = line.strip()

        if line[:5] == 'frame':
            detections.append(detections_frame)
            detections_frame = {}
            continue

        if line[:5] == 'track':
            track_id = line[line.find(':') + 1 : line.find(' ')]
            info = [float(item) for item in line[line.find(' ') + 1 : ].split(' ')]
            detections_frame[track_id] = info
            continue

    detections = detections[1:]
    
    return detections


# In[12]:


def centroid(coords):
    return [(coords[0] + coords[2]) / 2, (coords[1] + coords[3]) /2]


# In[13]:


def build_track_centroids_list(detections):
    track_centroids = {}

    for detections_frame in detections:   
        for track_id, coords in detections_frame.items():
            if not track_id in track_centroids.keys():
                track_centroids[track_id] = []
            else:
                track_centroids[track_id].append(centroid(coords))

    return track_centroids


# In[14]:


def linreg(pts):
    X = [coord[0] for coord in pts]
    Y = [coord[1] for coord in pts]
    N = len(X)
    if N < 2:
        return 0, 0
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    if det == 0:
        print(pts)
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det


# In[15]:


def build_track_trend(track_centroids_dict):
    linregs = []
    track_trends = {}

    for track_id, centroids in track_centroids_dict.items():
        if len(track_centroids_dict[track_id]) > 1:
            track_trends[track_id] = linreg(centroids)
    
    return track_trends


# In[16]:


def slope_to_degree(slope):
    return math.degrees(math.atan(slope))


# In[17]:


def filter_trends(track_trends, track_centroids, window_total_ratio, min_window_size, max_trend_slope_change):
    tracks_filtered = []
    trends_filtered = []

    for track_id, trend in track_trends.items():
        reliable = True

        trend_slope = slope_to_degree(track_trends[track_id][0])
        centroids = track_centroids[track_id]
        
        total = len(centroids)
        window = int(total * window_total_ratio)
        
        # Obj must be detected for more than total*window_total_ratio frames
        if window < min_window_size:
            reliable = False
        
        index = 0
        while reliable:
            if(index + window > len(centroids) - 1):
                break
                
            test_centroids = centroids[index: index + window - 1]            
            test_trend = linreg(test_centroids)
            test_trend_slope = slope_to_degree(test_trend[0])
            if abs(test_trend_slope - trend_slope) > max_trend_slope_change:
                '''
                print('track:'+track_id)
                print(test_trend_slope)
                print(trend_slope)
                print(abs(test_trend_slope - trend_slope))
                reliable = False
                '''
            index += window

        if reliable:
            tracks_filtered.append(track_id)
            trends_filtered.append(trend)

    return tracks_filtered, trends_filtered


# In[18]:


def plot_trends(tracks, trends):
    plt.figure(figsize=(16,9))

    plt.ylim(ymax=0)
    plt.ylim(ymin=720)
    plt.xlim(xmax=1280)
    plt.xlim(xmin=0)

    axes = plt.gca()
    x = np.array(axes.get_xlim())

    for s, i in trends[:]:
        plt.plot(x, s * x + i)

    plt.legend(tracks[:])

    plt.show()


# In[19]:


def plot_lanes(lanes):
    plt.figure(figsize=(16,9))

    plt.ylim(ymax=0)
    plt.ylim(ymin=720)
    plt.xlim(xmax=1280)
    plt.xlim(xmin=0)
    
    for lane in lanes:
        s = lane.slope
        i = lane.intersection
        x = np.array([lane.end_pts[0][0], lane.end_pts[1][0]])
        plt.plot(x, s * x + i)

    plt.show()


# In[20]:


def normalize_trend(trends, img_width, img_height):
    trends = np.array(trends)
    #trends[:, 1] = trends[:, 1] - img_width / 2 # move origin to bottom centre
    trends[:, 1] = trends[:, 1] / trends[:, 0] # norm. against slope
    trends[:, 0] = trends[:, 0] / trends[:, 1] # norm. against normalized intersection
    '''
    trends[:, 0] = trends[:, 0] / np.linalg.norm(trends[:, 0])
    trends[:, 1] = trends[:, 1] / np.linalg.norm(trends[:, 1]) #img_height 
    '''
    return trends


# In[21]:


### <- try use weighted average instead
def merge_trends_to_lanes(tracks, trends, track_centroids, max_slope_diff, max_x_intersection_diff):
    trends = np.array(trends)
    lanes = []
    
    lane = Lane(trends[0], [track_centroids[tracks[0]][0], track_centroids[tracks[0]][-1]])
    lanes.append(lane)
    
    is_new_lane = True
    for track, trend in zip(tracks, trends):
        for index, lane in zip(range(len(lanes)), lanes):
            if abs(slope_to_degree(trend[0]) - slope_to_degree(lane.slope)) < max_slope_diff:
            #and abs(trend[1]/trend[0] - lane.trend[1]/lane.trend[0]) / (abs(lane.trend[0]) + abs(lane.trend[0]))/2 < max_x_intersection_diff:
                lane.merge(trend, [track_centroids[track][0], track_centroids[track][-1]])
                lanes[index] = lane
                is_new_lane = False
                break
        if is_new_lane == True:
            lanes.append(Lane(trend, [track_centroids[track][0], track_centroids[track][-1]]))
            #print(abs(math.degrees(math.atan(trend[0])) - math.degrees(math.atan(lane[0]))))
            #print(abs(trend[1]/trend[0] - lane[1]/lane[0]) / (abs(lane[0]) + abs(trend[0]))/2)
        is_new_lane = True
    return lanes


# In[22]:


def find_track_bboxes(tracking_id, detections):
    bboxes = []
    frame_indices = []
    
    for detections_frame, frame_index in zip(detections, range(len(detections))):
        if tracking_id in detections_frame.keys():
            bboxes.append(detections_frame[tracking_id])
            frame_indices.append(frame_index)
            
    return bboxes, frame_indices


# In[23]:


def build_trackings(track_trends, track_centroids, lanes, detections):
    trackings = []
    
    track_ids = track_trends.keys()
    for tid in track_ids:
        bboxes, frames = find_track_bboxes(tid, detections)
        tracking = Tracking(tid, 
                            track_trends[tid], 
                            track_centroids[tid], 
                            bboxes, frames)
        tracking.associate_to_closest_lane(lanes)
        trackings.append(tracking)
        
    return trackings


# In[24]:


DETECTION_FILE_PATH = "./detection.txt"

IMG_W = 1280
IMG_H = 720

WINDOW_TOTAL_RATIO = 0.25
MIN_WINDOW_SIZE = 5
MAX_TREND_SLOPE_CHANGE = 20

MAX_SLOPE_DIFF = 7
MAX_X_INTERSECTION_DIFF = 60

detections = read_detection_file(DETECTION_FILE_PATH)
track_centroids = build_track_centroids_list(detections)
track_trends = build_track_trend(track_centroids)
tracks_filtered, trends_filtered = filter_trends(track_trends, track_centroids,
                                                 WINDOW_TOTAL_RATIO, 
                                                 MIN_WINDOW_SIZE, 
                                                 MAX_TREND_SLOPE_CHANGE)
#trends_normalized = normalize_trend(trends_filtered, IMG_W, IMG_H)
lanes = merge_trends_to_lanes(tracks_filtered, trends_filtered, track_centroids, MAX_SLOPE_DIFF, MAX_X_INTERSECTION_DIFF)
trackings = build_trackings(track_trends, track_centroids, lanes, detections)
for lane in lanes:
    lane.calc_average_bbox_change_ratio(trackings)
    lane.calc_average_bbox_speed(trackings)


