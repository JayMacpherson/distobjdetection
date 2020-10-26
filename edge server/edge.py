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

# global variables
zenith_inventor_stack = [[], []]

drone_worker = Queue()
worker_admin = Queue()
admin_drone = Queue()

lossy_encoding_threshold = 0.3
max_frame_per_time = 5
reply_time_threshold = 5
unstable_disconnection_count = 3
inconsistent_disconnection_count = 6
classification_time = 3
class_seq = {'satisfy': 1, 'poor': 6}


class Base(Process):
    """
    inherit Process and holds address
    """

    def __init__(self,
                 address,
                 _drone_worker=None,
                 _worker_admin=None,
                 _admin_drone=None
                 ):
        super().__init__()
        self.address = address
        self._drone_worker = _drone_worker
        self._worker_admin = _worker_admin
        self._admin_drone = _admin_drone


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


def get_workers():
    if not os.path.exists('worker_log.crp'):
        with open('worker_log.crp', 'wb') as _file_:
            _file_.write(dumps({}))
            return {}
    else:
        with open('worker_log.crp', 'rb') as _file_:
            workers = loads(_file_.read())
            print(workers)
            return workers


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
        proceed_stream = False

        while True:
            data = drone_sock.recv()

            piled_frame = self._drone_worker.qsize()
            if piled_frame > max_frame_per_time:
                drop = piled_frame - max_frame_per_time
                for _ in range(drop):
                    try:
                        self.remove(self._drone_worker.get())
                    except:
                        pass

            if proceed_stream:
                self.store(f'{cursor}drone', data)
                self._drone_worker.put(f'{cursor}drone')

                cursor += 1
            else:
                if not self._admin_drone.empty():
                    proceed_stream = True

            drone_sock.send(b"None")

    @staticmethod
    def store(serial, data: bytes):

        with open(f'file_array/{serial}', 'wb') as file:
            file.write(data)
        return

    @staticmethod
    def remove(serial: int):
        os.remove(f'file_array/{serial}')


class admin_packet_manager(Base):

    def run(self):

        ctx = zmq.Context()
        admin_sock = ctx.socket(zmq.REP)
        admin_sock.bind(self.address)
        self.cursor = 0

        proceed_stream = False
        dropped = 0

        last_frame_time = perf_counter()
        backlog = PriorityQueue()
        time_threshold = 0.05

        while True:
            _ = admin_sock.recv()

            if not proceed_stream:
                self._admin_drone.put(True)
                proceed_stream = True

            if not self._worker_admin.empty():

                params = self._worker_admin.get()

                frame_id = params['frame_id']
                params['worker']['dropped_frames'] = 0

                if frame_id > self.cursor + 1:
                    # stores the frame temp
                    backlog.put((params['frame_id'], params))

                    if (perf_counter() - last_frame_time) > time_threshold:
                        new_frame = backlog.get()[1]
                        new_id = new_frame['frame_id']
                        dropped += (new_id - (self.cursor + 1))

                        self.pack_send(new_frame, dropped, admin_sock)

                        dropped = 0
                        # updating the counter
                        last_frame_time = perf_counter()
                    else:
                        admin_sock.send(b'None')
                        continue

                elif frame_id == self.cursor + 1:
                    self.cursor = frame_id
                    self.pack_send(params, dropped, admin_sock)
                    dropped = 0
                    last_frame_time = perf_counter()

                else:
                    # this can only happen when a frame comes really late
                    # and it is to be forgotten but does not update the cursor
                    admin_sock.send(b'None')
                    self.remove(frame_id)
                    last_frame_time = perf_counter()

            else:
                admin_sock.send(b'None')

    @staticmethod
    def get(serial: int):

        with open(f'file_array/{serial}', 'rb') as file:
            data = file.read()

        return serialize(data)

    def pack_send(self, frame, dropped, admin_client):
        # getting data from file
        data = self.get(frame['frame_id'])

        frame['worker']['dropped_frames'] = dropped

        # serializing data
        data = serialize({
            'detection_data': frame['detection_data'],
            'raw_frame': data,
            'worker': frame['worker']},
            entry=1
        )

        admin_client.send(data)

        self.cursor = frame['frame_id']
        self.remove(self.cursor)

    @staticmethod
    def remove(serial: int):
        os.remove(f'file_array/{serial}')


class workers_packet_manager(Base):

    def run(self):

        ctx = zmq.Context()
        worker_sock = ctx.socket(zmq.REP)
        worker_sock.bind(self.address)

        self.workers = get_workers()
        self.workers_data = {}
        self.connection = {}

        worker_log = []
        cursor = 0

        while True:
            data = worker_sock.recv()
            data = loads(data)
            status = ''

            entry = data.get('entry', None)
            worker_id = data.get('id', None)

            if entry and worker_id:
                if entry == 'connect':
                    status = self.classify_worker(worker_id, False)
                    data = b'None'
                elif entry == 'none':
                    data = b'None'
                else:
                    status = self.classify_worker(worker_id, True)
                    data = data.get('data')

                if data != b'None':
                    data = serialize(data)

                    worker_name = data[3]
                    speed = data[1]
                    frame_id = data[0]
                    detection_data = data[2]

                    worker_log = {
                        'worker_name': worker_name,
                        'worker_speed': speed,
                        'lossy_encoding': False,
                        'status': status
                    }

                    if data[1] > lossy_encoding_threshold:
                        if worker_log:
                            worker_log['lossy_encoding'] = True

                    self._worker_admin.put({
                        'frame_id': frame_id,
                        'detection_data': detection_data,
                        'worker': worker_log
                    })

                if not self._drone_worker.empty():

                    serial = self._drone_worker.get()
                    _data = self.get(serial)

                    os.rename(f'file_array/{serial}', f'file_array/{cursor}')
                    serial = cursor
                    cursor += 1

                    if data[1] > lossy_encoding_threshold:
                        _data = self.lossy_encoding(_data)

                        if worker_log:
                            worker_log['lossy_encoding'] = True

                    worker_sock.send(serialize([serial, _data], entry=1))
                    # update worker inspector
                    # identity = None if data == b'None' else data[4]
                    # status = self.worker_inspector(identity)
                    #
                    # if status != 'Inconsistent':
                    #     worker_sock.send(serialize([serial, _data, status], entry=1))
                    # else:
                    #     worker_sock.send(serialize([b'None', b'None', status], entry=1))

                else:
                    worker_sock.send(b'None')
            else:
                worker_sock.send(b'None')

    def classify_worker(self, worker_id, valid_data: bool):
        if worker_id in self.workers_data.keys():
            worker = self.workers_data[worker_id]

            if perf_counter() - worker['last_updated'] > classification_time*60:
                worker['disconnections'] = 0
                worker['last_updated'] = perf_counter()

            if not valid_data:
                worker['disconnections'] += 1

            count = worker['disconnections']

            if count > class_seq['poor']:
                return 'Poor'
            elif count > class_seq['satisfy']:
                return 'Satisfactory'
            else:
                return 'Excellent'

        else:
            self.workers_data[worker_id] = {
                'disconnections': 0,
                'last_updated': perf_counter()
            }
            return 'Excellent'

    @staticmethod
    def lossy_encoding(data):
        # msg, frame = extract_frame_from_tensor(data)

        # reducing frame quality by half
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        #
        # data = prepare_frame_tensor(frame, msg)
        return data

    def worker_inspector(self, identity):
        if identity:
            old = deepcopy(self.workers)
            if identity not in self.workers.keys():
                status = 0
                self.workers[identity] = status
            else:
                status = self.workers[identity]

            for _identity_, last_updated in self.connection.items():
                if (perf_counter() - last_updated) > reply_time_threshold:
                    self.workers[_identity_] += 1
                    if identity == _identity_:
                        status += 1

            self.connection[identity] = perf_counter()

            if old != self.workers:
                with open('worker_log.crp', 'wb') as _file_:
                    _file_.write(dumps(self.workers))

            if status > inconsistent_disconnection_count:
                _status_ = 'Inconsistent'
            elif status > unstable_disconnection_count:
                _status_ = 'Unstable'
            else:
                _status_ = 'Stable'

            return _status_
        else:
            return 'None'

    @staticmethod
    def get(serial: int):

        with open(f'file_array/{serial}', 'rb') as file:
            data = file.read()

        return data


class drone_admin(Base):
    def run(self):
        global zenith_inventor_stack

        ctx = zmq.Context()
        admin_sock = ctx.socket(zmq.REP)
        admin_sock.bind(self.address)

        while True:
            data = admin_sock.recv()
            msg, feedback = extract_valid(data)

            if msg == 'Detector':

                if feedback:
                    # queues if detected frames comes back  from detector
                    zenith_inventor_stack[1] += [data]

                # checks if there is frame in the queue
                if zenith_inventor_stack[0]:

                    # sends frame to detector
                    admin_sock.send(zenith_inventor_stack[0][0])

                    # de-queues
                    del zenith_inventor_stack[0][0]
                else:
                    # sends none to detector if no frames are available in queue
                    admin_sock.send(b'None')
            # this is for FS
            else:
                # queues frame in stack for detector to use
                zenith_inventor_stack[0] += [data]

                # checks if there is return detected frames from detector
                if zenith_inventor_stack[1]:
                    # sends detected frames to FS
                    admin_sock.send(zenith_inventor_stack[1][0])
                    # de-queues
                    del zenith_inventor_stack[1][0]
                else:
                    # sends None if no detected frames are available
                    admin_sock.send(b'None')


if __name__ == '__main__':
    try:
        print('[INFO] Cleaning directory...')
        shutil.rmtree('file_array')
        os.mkdir('file_array')

    except:
        os.mkdir('file_array')

    ZI = drone_admin("tcp://*:4001")
    ZI.start()
    AD = admin_packet_manager(
        "tcp://*:4022",
        _worker_admin=worker_admin,
        _admin_drone=admin_drone
    )

    AD.start()
    WM = workers_packet_manager(
        "tcp://*:4033",
        _drone_worker=drone_worker,
        _worker_admin=worker_admin
    )

    WM.start()
    DM = Drones_packet_manager(
        "tcp://*:4044",
        _drone_worker=drone_worker,
        _admin_drone=admin_drone
    )

    DM.start()
    print('Edge Server Active')

    AD.join()