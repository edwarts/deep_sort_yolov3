#  connect to Kafka
import sys

from kafka import SimpleProducer, KafkaClient,KafkaConsumer
import cv2
import time
import numpy as np
from PIL import Image
kafka = KafkaClient('localhost:9092')
producer = SimpleProducer(kafka)
# Assign a topic
topic = 'test-video-streaming'


def video_emitter(video):
    # Open the video
    video = cv2.VideoCapture(video)
    print(' emitting.....')

    # read the file
    frame_index=0
    while (video.isOpened):
        # read the image in each frame
        success, image = video.read()

        # check if the file has read the end
        if not success:
            break

        # convert the image png
        ret, jpeg = cv2.imencode('.jpeg', image)
        # Convert the image to bytes and send to kafka
        out_put_emit_data=jpeg.tobytes()
        print(len(np.array(jpeg)))
        # out_put_emit_data = np.array(image).tobytes() too big error
        print(out_put_emit_data)
        producer.send_messages(topic,out_put_emit_data)
        # To reduce CPU usage create sleep time of 0.2sec
        time.sleep(0.2)
        frame_index+=1
        # for test purpose, use 5 frame
        if frame_index==5:
            break
    # clear the capture
    video.release()
    print('done emitting')

def video_consumer(topic):
    consumer = KafkaConsumer(topic, group_id='view',bootstrap_servers = ['127.0.0.1:9092'])
    for msg in consumer:
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + msg.value + b'\r\n\r\n')


if __name__ == '__main__':
    video_emitter(sys.argv[1])