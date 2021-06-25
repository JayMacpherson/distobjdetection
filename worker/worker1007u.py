import zmq
import pickle
import numpy as np
import cv2
import OD
from random import randint
from time import perf_counter, sleep
import os
from pickle import loads, dumps
from PIL import Image
import schedule

scheduled_time = 15
chance = 25
sleep_amount = 10
identity = 1007

socket = zmq.Context()
socket = socket.socket(zmq.REQ)
socket.connect("tcp://10.123.1.60:4033")
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

    answer = None
    while answer not in ("y", "n"):
        answer = input("Do you want to mess with default deadlines y/n: ")
        if answer == "y":
            global scheduled_time
            global chance
            global sleep_amount
            scheduled_time = int(input('Select schedule in seconds '))
            chance = int(input('Select probability of sleep (1 - 100) '))
            sleep_amount = int(input('Select duration of sleep in seconds '))
        elif answer == "n":
            break
        else:
            print("Please enter y or n.")

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


def scheduled_sleep():
    roll = randint(1, 100)
    if roll <= chance:
        print('sleeping')
        sleep(sleep_amount)
        print('awake')


user = pre_launch()
_id_ = user.data['id']
_username_ = user.data['username']
_status_ = user.data['status']

print('[INFO] Worker is Active...')

schedule.every(scheduled_time).seconds.do(scheduled_sleep)

while True:
    schedule.run_pending()
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
