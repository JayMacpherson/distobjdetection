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

default_latency = 0

socket = zmq.Context()
socket = socket.socket(zmq.REQ)
socket.connect("tcp://localhost:4033")
data = None

print('[INFO] Loading detector...')
detector = OD.Tensorflow()


def serialize(content, entry=0):  # entry at 1 means convert list to byte
    try:
        if entry:
            return pickle.dumps(content)
        else:
            return pickle.loads(content)
    except:
        return False


def extract_frame_from_tensor(data_):
    """
    To extract image and message extracted from zmq network
    :param data_: data to extract
    :return: message and image
    """

    msg_, type_, shape, image_, feedback = serialize(data_)
    extracted_frame = np.frombuffer(image_, dtype=type_)
    return msg_, extracted_frame.reshape(shape)


def end():
    print('You are now inconsistent in your connection with this server')
    print('Therefore banning you as a worker')

    identity = {'id': my_id, 'username': _username_, 'status': 'Inconsistent'}

    with open('log.crp', 'wb') as file:
        file.write(dumps(identity))

    exit()


def pre_launch(force: bool = False):
    if not os.path.exists('log.crp') or force:
        _id_ = randint(11111111, 999999999)
        username = input('Input your username: ')
        status = 'Stable'

        if not username:
            username = str(randint(1000, 9999))

        identity = {'id': _id_, 'username': username, 'status': status}

        with open('log.crp', 'wb') as file:
            file.write(dumps(identity))
        return _id_, username, status
    else:
        with open('log.crp', 'rb') as file:
            identity = loads(file.read())

        _id_ = identity.get('id', None)
        username = identity.get('username', None)
        status = identity.get('status', None)

        if not _id_ or not username or not status:
            pre_launch(force=True)
        else:
            if status == 'inconsistent':
                print('Oops, seems you have been inconsistent according to your history')
                print('So you are no longer allowed to work for this server')
                exit()
            elif status == 'Unstable':
                print('Its seems you are becoming unstable so be careful')

            return _id_, username, status


def prepare_frame_tensor(frame, text):
    """
    This function prepares the image to be transfer over zmq network
    :param frame: image to prepare
    :param text: message to be attached
    :return: prepared data
    """

    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    data_ = [text,
             str(frame.dtype),
             frame.shape,
             frame,
             False]
    return serialize(data_, entry=1)


my_id, _username_, _status_ = pre_launch()
print('[INFO] Worker is Active...')
while True:
    if not data:
        data = {'entry': 'connect', 'id': my_id}
    elif data == b'None':
        data = {'entry': 'none', 'id': my_id}
    else:
        data = {'entry': 'payload', 'data': data, 'id': my_id}

    socket.send(dumps(data))

    data = socket.recv()

    if data != b'None':
        serial, data = serialize(data)

        # if status == 'Inconsistent':
        #     end()
        # elif status == 'Unstable':
        #     if _status_ != 'Unstable':
        #         with open('log.crp', 'wb') as file:
        #             file.write(dumps({
        #                 'id': my_id,
        #                 'username': _username_,
        #                 'status': status
        #             }))
        #         print('Its seems you are becoming unstable so be careful')
        #         _status_ = status

        data_, image = extract_frame_from_tensor(data)

        time = perf_counter()
        data = detector.detection(image, data_)
        sleep(default_latency)

        speed = perf_counter() - time

        data = serialize([serial, speed, data, _username_, my_id], entry=1)
