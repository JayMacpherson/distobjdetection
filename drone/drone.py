import os
import zipfile
import shutil
import zmq
from time import sleep, time
from data import extract, prepare
from pickle import *
from PIL import Image, UnidentifiedImageError
import numpy as np
from threading import Thread
from tqdm import tqdm
from pickle import dumps, loads
from _pickle import UnpicklingError


results = []
stream = []
chosen = None
chosen2 = None
latency = 0.03125
PRE_CONVERSION = True
conversion_progress = 0
CONVERSION_THREADS = 14


def prepare_image(image: str):
    try:
        image = Image.open(image)
        (im_width, im_height) = image.size
        return list(np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8))
    except UnidentifiedImageError:
        with open(image, 'rb') as _file_:
            return loads(_file_.read())


def choose(_list_: list) -> str:
    # printing filtered to be selected
    [print(f'{i}: {title}') for i, title in enumerate(_list_)]
    ind = 0

    while True:
        try:
            ind = int(input('Select index: ').strip())
            if ind > len(_list_)-1 or ind < 0:
                raise ValueError
            break
        except ValueError:
            print('You entered invalid response!')
            continue

    return _list_[ind]


def pre_stream_conversions(data_pack: list, ind: int, bar, last: bool = False, launch: bool = True):
    global conversion_progress

    if launch:
        thread = Thread(
            target=pre_stream_conversions,
            args=[data_pack, ind, bar, last, False],
            daemon=True
        )
        thread.start()
    else:
        for item in data_pack:
            numpy_array = prepare_image(f'todo/{chosen}/{chosen}/{chosen2}/img/{item}')
            with open(f'todo/{chosen}/{chosen}/{chosen2}/img/{item}', 'wb') as _file_:
                _file_.write(dumps(numpy_array))
            bar.update(1)

        conversion_progress += 1
        # blocking if it is the last thread that must finish
        if last:
            while conversion_progress != CONVERSION_THREADS:
                sleep(1)
        else:
            exit()


# showing list of video zip sections
if not os.path.exists('todo'):
    print('Take folder called "todo" and put it in drone directory!')
    exit()
else:
    file_list = os.listdir('todo')
    chosen = choose(file_list).split('.')[0]

    if not os.path.exists(f'todo/{chosen}'):
        print('Unzipping parent folder once...')

        with zipfile.ZipFile(f'todo/{chosen}.zip') as zf:
            for member in tqdm(zf.infolist(), desc='Extracting '):
                try:
                    zf.extract(member, f'todo/{chosen}')
                except zipfile.error as e:
                    pass

        print('Unzip complete')

    file_list = os.listdir(f'todo/{chosen}/{chosen}')
    chosen2 = choose(file_list)

    # acquiring result of the stream to memory
    with open(f'todo/{chosen}/{chosen}/{chosen2}/groundtruth.txt') as file:
        results = file.read().splitlines()

    stream = os.listdir(f'todo/{chosen}/{chosen}/{chosen2}/img')

    # determining if the files in folder is already converted in numpy array
    if stream:
        with open(f'todo/{chosen}/{chosen}/{chosen2}/img/{stream[-1]}', 'rb') as test:
            try:
                loads(test.read())
                converted = True
            except UnpicklingError:
                converted = False


if not converted and PRE_CONVERSION:
    # prepare frames
    segment_number = int(len(stream)/CONVERSION_THREADS)
    split_stream = [stream[i:segment_number+i] for i in range(0, len(stream), segment_number)]

    # if extra rack is left it's gets added up with the last rack
    if len(split_stream) > CONVERSION_THREADS:
        split_stream[-2] += split_stream[-1]
        del split_stream[-1]

    with tqdm(total=len(stream)) as bar:
        for i in range(CONVERSION_THREADS-1):
            pre_stream_conversions(split_stream[i], i+1, bar)

        pre_stream_conversions(split_stream[-1], CONVERSION_THREADS, bar, last=True, launch=False)

# connects to the edge server
socket = zmq.Context()
socket = socket.socket(zmq.REQ)

socket.connect("tcp://10.123.1.60:4044")
print('[INFO] Stream has started!')

for i, frame in enumerate(tqdm(stream, desc='Stream: ')):
    mark = time()

    # reads from file
    if not PRE_CONVERSION:
        frame_byte = prepare_image(f'todo/{chosen}/{chosen}/{chosen2}/img/{frame}')
    else:
        with open(f'todo/{chosen}/{chosen}/{chosen2}/img/{frame}', 'rb') as file:
            frame_byte = loads(file.read())

    frame_result = results[i]

    data = dumps({
        'frame': frame_byte,
        'result': frame_result,
        'origination': [chosen, chosen2]
    })

    # delays clock minus computational time cost
    computational_time = time()-mark
    if latency > computational_time:
        sleep(latency-computational_time)

    # sends prepared data
    socket.send(data)

    # receives feedback from edge server
    _ = socket.recv()

print('stream finished')