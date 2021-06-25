import zmq
import pickle
from multiprocessing import Process, Queue
import numpy as np
import cv2
import os
import shutil
from pickle import loads, dumps
from queue import PriorityQueue
from time import perf_counter
from copy import deepcopy
from time import sleep, time
from threading import Thread
import json

# global variables
zenith_inventor_stack = [[], []]

drone_worker = Queue()
worker_admin = Queue()
admin_drone = Queue()

worker_tolerance = 3
frame_wait_tolerance = 2
class_seq = {'Satisfactory': 2, 'Poor': 3}

class Base(Process):
    """
    inherit Process and holds address
    """

    def __init__(self,
                 address,
                 _drone_worker=None,
                 _worker_admin=None,
                 _admin_drone=None,
                 _stream_tolerance=0.2
                 ):
        super().__init__()
        self.address = address
        self._drone_worker = _drone_worker
        self._worker_admin = _worker_admin
        self._admin_drone = _admin_drone
        self.connections = None
        self.workers = None
        self.incoming_stack = []
        self.colliding_agents = {}
        self.stream_tolerance = _stream_tolerance
        self.total_iou = []  # total number of iou


class File:
    def __init__(self, filename: str):
        self.filename = filename
        try:
            self.read = open(f'{self.filename}.crp', 'rb')
            self.data = loads(self.read.read())
        except FileNotFoundError:
            with open(f'{self.filename}.crp', 'wb') as file:
                file.write(pickle.dumps({}))
            self.data = {}

    def save(self):
        with open(f'{self.filename}.crp', 'wb') as file:
            file.write(dumps(self.data))


def serialize(list_, entry=0):
    try:
        if entry:
            return pickle.dumps(list_)
        else:
            return pickle.loads(list_)
    except:
        return False


def extract_valid(data):
    msg, type_, shape, image, feedback = serialize(data)
    return msg, feedback


def extract_frame_from_tensor(data_):
    """
    To extract image and message extracted from zmq network
    :param data_: data to extract
    :return: message and image
    """

    msg_, type_, shape, image_, feedback = serialize(data_)
    extracted_frame = np.frombuffer(image_, dtype=type_)
    return msg_, extracted_frame.reshape(shape)


def prepare_frame_tensor(frame, text: str, raw: bool = False):
    """
    This function prepares the image to be transfer over zmq network
    :param frame: image to prepare
    :param text: message to be attached
    :param raw: bool that is true when the excepted return should image tensor
    :return: prepared data
    """

    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    data_ = [text,
             str(frame.dtype),
             frame.shape,
             frame,
             False]
    if not raw:
        return serialize(data_, entry=1)
    else:
        return data_


class Drones_packet_manager(Base):

    def run(self):
        ctx = zmq.Context()
        drone_sock = ctx.socket(zmq.REP)
        drone_sock.bind(self.address)

        cursor = 0

        while True:
            data = drone_sock.recv()

            self.store(cursor, data)
            self._drone_worker.put([cursor, time()])

            cursor += 1

            drone_sock.send(b"None")

    @staticmethod
    def store(serial, data: bytes):
        with open(f'file_array/{serial}', 'wb') as file:
            file.write(data)
        return

    @staticmethod
    def remove(serial: int):
        os.remove(f'file_array/{serial}')


class workers_packet_manager(Base):

    def run(self):

        ctx = zmq.Context()
        worker_sock = ctx.socket(zmq.REP)
        worker_sock.bind(self.address)
        self.workers = File('worker_log')
        self.connections = {}

        arriving_cursor = -1
        awaiting = PriorityQueue()
        waiting_countdown = perf_counter()
        dropped_counter = 0

        thread = Thread(target=self.connection_clock, daemon=True)
        thread.start()



        while True:
            # unloading to stack
            for _ in range(self._drone_worker.qsize()):
                self.incoming_stack.append(self._drone_worker.get())

            data = pickle.loads(worker_sock.recv())
            user_id = data['user_id']
            username = data['username']

            if user_id not in self.workers.data.keys():
                self.workers.data[user_id] = 0
                worker_status = 'Excellent'
            else:
                worker_status = self.manage_workers(user_id)

            if worker_status != 'Poor':
                if data['content']:
                    awaiting.put((data['serial'], data))

                if self.incoming_stack:
                    new_data_cursor = self.incoming_stack.pop(0)

                    if not awaiting.empty():
                        first = awaiting.get()
                        present_incoming = None
                        inspecting_packet = new_data_cursor if not self.incoming_stack else self.incoming_stack[-1]
                        #  figuring out if the latest incoming packet from drone is new or not
                        if (time() - inspecting_packet[1]) < self.stream_tolerance:
                            present_incoming = inspecting_packet[0]

                        if present_incoming is not None:
                            if first[0] > arriving_cursor:
                                if (perf_counter() - waiting_countdown) > frame_wait_tolerance:
                                    waiting_countdown = perf_counter()
                                    arriving_cursor = first[0] + 1
                                    self.calculate_accuracy_difference(present_incoming, first[1])
                                else:
                                    awaiting.put(first)

                            elif first[0] == arriving_cursor:
                                waiting_countdown = perf_counter()
                                arriving_cursor += 1
                                self.calculate_accuracy_difference(present_incoming, first[1])
                            else:
                                dropped_counter += 1
                                print(f'{dropped_counter}: total dropped')

                    new_data_cursor = new_data_cursor[0]

                else:
                    new_data_cursor = None

                if user_id not in self.connections.keys():
                    print(f'{username} came online')
                    self.connections[user_id] = [7, 7, username,
                                                 new_data_cursor]  # 4 is the max second excepted for a worker to reply
                    # while 5 is the countdown sec

                else:
                    connect_data = self.connections[user_id]

                    if data['content']:
                        max_sec = max([data['speed'] + worker_tolerance, connect_data[1]])
                        max_sec = data['speed'] + worker_tolerance if max_sec == 4 else max_sec
                    else:
                        max_sec = connect_data[1]

                    self.connections[user_id] = [max_sec, max_sec, username, new_data_cursor]

                if new_data_cursor is not None:
                    data = self.get(new_data_cursor)
                else:
                    data = b'None'
            else:
                new_data_cursor = None
                data = b'None'

            data = {
                'status': worker_status,
                'data': data,
                'serial': new_data_cursor
            }

            worker_sock.send(pickle.dumps(data))

    def connection_clock(self):
        while True:
            try:
                sleep(.1)
                for user_id, data in self.connections.items():
                    countdown = data[0] - 0.1
                    if countdown < 0:
                        if data[3] is not None:
                            print(f'{data[2]}: missed the deadline')

                            crimes = self.workers.data[user_id]
                            self.workers.data[user_id] = crimes + 1
                            self.workers.save()
                        else:
                            print(f'{data[2]}: went offline')

                        del self.connections[user_id]
                    else:
                        self.connections[user_id] = [countdown, data[1], data[2], data[3]]
            except RuntimeError:
                continue

    def manage_workers(self, user_id):
        crimes = self.workers.data[user_id]

        if crimes > class_seq['Poor']:
            return 'Poor'
        elif crimes > class_seq['Satisfactory']:
            return 'Satisfactory'
        else:
            return 'Excellent'

    def calculate_accuracy_difference(self, incoming, arrived):
        incoming_data = pickle.loads(self.get(incoming))
        origination = arrived['origination']

        # checking if origination colliding detections co-respond
        if incoming_data['origination'] == origination:
            if not os.path.exists(origination[0]):
                os.mkdir(origination[0])

            if origination[1] not in self.colliding_agents.keys():
                self.colliding_agents[origination[1]] = []

            if arrived['new_result']:
                detection = None
                for item in arrived['new_result']:
                    if item[-1].lower() in origination[1]:
                        detection = item[0]
                        break
                incoming_data = [int(item) for item in incoming_data['result'].split(',')]
                if detection:
                    iou = self.bb_intersection_over_union(incoming_data, detection)

                else:
                    iou = 0

                data = {
                    'current_frame': incoming,
                    'groundtruth': incoming_data,
                    'detection_id': arrived['serial'],
                    'result': detection,
                    'iou': iou
                }

                self.colliding_agents[origination[1]].append(data)

                with open(f'{origination[0]}/{origination[1]}.json', 'wb') as file:
                    file.write(pickle.dumps(self.colliding_agents[origination[1]]))

                print(f'{incoming} -> {arrived["serial"]}')
                print(f'{iou}')
                self.total_iou.append(iou)
                if self.total_iou:
                    print('Average IOU', sum(self.total_iou) / len(self.total_iou))


    def bb_intersection_over_union(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # box A coordinates
        Ax1 = boxA[0]
        Ay1 = boxA[1]
        Ax2 = boxA[0] + boxA[2]
        Ay2 = boxA[1] + boxA[3]
        Aw = boxA[2]
        Ah = boxA[3]

        # box B coordinates
        Bx1 = boxB[0]
        By1 = boxB[1]
        Bx2 = boxB[0] + boxB[2]
        By2 = boxB[1] + boxB[3]
        Bw = boxB[2]
        Bh = boxB[3]

        left = max(Ax1, Bx1)
        right = min(Ax2, Bx2)
        bottom = min(Ay2, By2)
        top = max(Ay1, By1)

        # compute the intersection over union IOU

        if (left < right) and (top < bottom):
            boxAArea = Aw * Ah
            boxBArea = Bw * Bh
            interArea = (right - left) * (bottom - top)
            iou = interArea / float(boxAArea + boxBArea - interArea)

            return iou

        else:
            return 0

    @staticmethod
    def get(serial: int):
        with open(f'file_array/{serial}', 'rb') as file:
            data = file.read()

        return data


if __name__ == '__main__':

    try:
        print('[INFO] Cleaning directory...')
        shutil.rmtree('file_array')
        os.mkdir('file_array')

    except:
        os.mkdir('file_array')

    stream_tolerance = float(input("stream tolerance ") or 0.2)
    WM = workers_packet_manager(
        "tcp://*:4033",
        _drone_worker=drone_worker,
        _worker_admin=worker_admin,
        _stream_tolerance=stream_tolerance
    )

    WM.start()
    DM = Drones_packet_manager(
        "tcp://*:4044",
        _drone_worker=drone_worker,
        _admin_drone=admin_drone
    )

    DM.start()
    print('Edge Server Active')

    DM.join()