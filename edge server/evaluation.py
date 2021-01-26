import pickle
from _pickle import UnpicklingError
import os


def choose(_list_: list) -> str:
    # printing filtered to be selected
    [print(f'{i}: {title}') for i, title in enumerate(_list_)]
    ind = 0

    while True:
        try:
            ind = int(input('Select index: ').strip())
            if ind > len(_list_) - 1 or ind < 0:
                raise ValueError
            break
        except ValueError:
            print('You entered invalid response!')
            continue

    return _list_[ind]


path = ''

while True:
    directory = os.listdir(path if path else None)
    path = f'{path}/{choose(directory)}'.strip('/')

    if os.path.isdir(path):
        continue
    else:
        if path.endswith('.json'):
            try:
                with open(path, 'rb') as file:
                    data = pickle.loads(file.read())
                    print(data)

                    iou_list = [item['iou'] for item in data]
                    iou_list = list(filter(None, iou_list))
                    latency_list = [item['current_frame'] - item['detection_id'] for item in data]
                    # dead frame number
                    dead_frames = len(data) - len(iou_list)
                    average_latency = sum(latency_list) / len(latency_list)
                    average_iou = sum(iou_list) / len(iou_list)

                    print(f'Average IOU, deadframes, latency is {average_iou, dead_frames, average_latency}')
                exit()
            except UnpicklingError:
                print('Hmmm something is wrong')
        else:
            print('This script is not written to recognize this file')
            exit()

