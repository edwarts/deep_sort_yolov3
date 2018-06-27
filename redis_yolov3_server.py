from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import flask
from threading import Thread
import base64
import redis
import uuid
import json
import io
from keras.preprocessing.image import img_to_array


warnings.filterwarnings('ignore')

IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"
 
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 1
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25


app = flask.Flask(__name__)
db = redis.StrictRedis(host="localhost", port=6379, db=0)

    
def base64_encode_image(a):
    return base64.b64encode(a).decode("utf-8")


def base64_decode_image(a, dtype, shape):
    if sys.version_info.major == 3:
        a = bytes(a, encoding="utf-8")
    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    return a


def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = img_to_array(image)
    return image


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image)
            image = image.copy(order="C")

            k = str(uuid.uuid4())
            d = {"id": k, 
                 "image": base64_encode_image(image),
                 "h": image.shape[0],
                 "w": image.shape[1]
                }
            db.rpush(IMAGE_QUEUE, json.dumps(d))
            
            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(CLIENT_SLEEP)
            data["success"] = True

    return flask.jsonify(data)


@app.route('/explore/<cam_id>', methods=['GET'])
def explore(cam_id):
    data = {"success": False}

    if flask.request.method == "GET":
        try:
            with open(cam_id) as jfile:
                cam_data = json.load(jfile)
                data["cam_data"] = cam_data
                data["success"] = True
        except Exception as e:
            data["error"] = str(e)
    return flask.jsonify(data)


def detect_process():   
    model = YOLO()
    
    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = base64_decode_image(q["image"], IMAGE_DTYPE,
                (BATCH_SIZE, q["h"], q["w"], IMAGE_CHANS))

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            imageIDs.append(q["id"])
        
        if len(imageIDs) > 0:
            print("* Batch size: {}".format(batch.shape))
            for imageID, image in zip(imageIDs, batch):
                image = Image.fromarray(image.astype("uint8"))
                boxs = model.detect_image(image)

                output = []
                for b in boxs:
                    r = {"xywh": [b[:4]],
                         "score": 1*b[4],
                         "class": 1*b[5]+0.0}
                    output.append(r)
                    
                db.set(imageID, json.dumps(output))
                db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(SERVER_SLEEP)
        
        
if __name__ == "__main__":
    print("* Starting model service...")
    t = Thread(target=detect_process, args=())
    t.daemon = True
    t.start()

    print("* Starting web service...")
    app.run()