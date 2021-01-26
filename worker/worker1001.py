import zmq
import pickle
import numpy as np
import cv2
import OD
from random import randint
from time import perf_counter, sleep
import os
from random import randint
from pickle import loads, dumps
from PIL import Image

default_latency = 0
identity = 1001

socket = zmq.Context()
socket = socket.socket(zmq.REQ)
socket.connect("tcp://localhost:4033")
data = b'None'

print('[INFO] Loading detector...')
detector = OD.Tensorflow()


class File:
    def __init__(self, filename: str):
        self.filename = filename
        try:
            self.read = open(f'{self.filename}.crp', 'rb')
            self.data = loads(self.read.read())
        except FileNotFoundError:
            pass

    def save(self):
        with open(f'{self.filename}.crp', 'wb') as file:
            file.write(dumps(self.data))


def pre_launch(force: bool = False):
    global identity

    if not os.path.exists(f'log_{identity}.crp') or force:
        username = input('Input your username: ')
        status = 'Excellent'

        if not username:
            username = str(randint(1000, 9999))
        pack = {'id': identity, 'username': username, 'status': status}

        with open(f'log_{identity}.crp', 'wb') as file:
            file.write(dumps(pack))

        return File(f'log_{identity}')
    else:
        file = File(f'log_{identity}')

        _id_ = file.data.get('id', None)
        username = file.data.get('username', None)
        status = file.data.get('status', None)

        if not _id_ or not username or not status:
            pre_launch(force=True)
        else:
            if status == 'Poor':
                print('Oops, seems you have been inconsistent according to your history')
                print('So you are no longer allowed to work for this server')
                exit()
            elif status == 'Satisfactory':
                print('Be careful of further missed deadlines')

            return file


user = pre_launch()
_id_ = user.data['id']
_username_ = user.data['username']
_status_ = user.data['status']

print('[INFO] Worker is Active...')

while True:
    if data == b'None':
        data = dumps({
            'username': _username_,
            'user_id': _id_,
            'content': False
        })

    socket.send(data)

    data = socket.recv()

    data = loads(data)

    worker_status = data['status']

    if _status_ != worker_status:
        _status_ = worker_status
        user.data['status'] = worker_status
        user.save()

        if _status_ == 'Satisfactory':
            print('Performing satisfactorily')

    if worker_status == 'Poor':
        print('You are now banned for inconsistency')
        exit()

    if data['serial'] is not None:
        main = loads(data['data'])
        frame_byte = main['frame']

        # actual detection
        time = perf_counter()
        frame_detection = detector.detection(frame_byte, None)
        sleep(default_latency)
        speed = perf_counter() - time

        scaled_detections = []

        # scaling detections
        if frame_detection:
            frame_byte = np.asarray(frame_byte)
            height, width = frame_byte.shape[:2]

            for box in frame_detection:
                box, accuracy, name = box

                # calculating the object position and labeling it
                ymin, xmin, ymax, xmax = box
                try:
                    h = int((ymax - ymin) * height)
                    w = int((xmax - xmin) * width)
                    y = int((height * ymax) - h)
                    x = int((width * xmax) - w)
                    scaled_detections.append(((x, y, w, h), accuracy, name))
                except:
                    pass

        data = dumps({
            'serial': data['serial'],
            'speed': speed,
            'new_result': scaled_detections,
            'username': _username_,
            'user_id': _id_,
            'origination': main['origination'],
            'content': True
        })
        print(scaled_detections)
    else:
        data = b'None'
