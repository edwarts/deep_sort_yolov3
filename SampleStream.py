#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import argparse
import cv2 as cv
import os
from timeit import time
import warnings
import sys
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description='Sample a video stream.')
parser.add_argument('input_parth', help='Path to input video stream.')
parser.add_argument('output_path', help='Path to output video stream.')
parser.add_argument('start', type=int, help='Start frame. If unset, start from beginning.')
parser.add_argument('end', type=int, help='End frame. If unset, run till the last frame.')


warnings.filterwarnings('ignore')

# %%
def _main(args):
    input_parth = args.input_parth
    output_path = args.output_path
    start = args.start
    end = args.end
    
    cap = cv.VideoCapture(input_parth)
    w = int(cap.get(3))
    h = int(cap.get(4))
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    fps = cap.get(cv.CAP_PROP_FPS)
    
    out = cv.VideoWriter(output_path, fourcc, fps, (w, h))
    
    index = 0
    while True:
        if end != None and index > end:
            break
        
        ret, frame = cap.read()
        if ret != True:
            break
        
        if index >= start:
            out.write(frame)
        
        index += 1

    cap.release()
    out.release()


if __name__ == '__main__':
    _main(parser.parse_args())