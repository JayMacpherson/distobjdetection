import pickle
from _pickle import UnpicklingError
import os
import csv

# field names
fields = ['Stream_frame_id', 'IOU']

# name of csv file
filename = "results_IOU.csv"


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
                    rows = [[item['current_frame'], item['iou']] for item in data]
                    iou_list = [item['iou'] for item in data]
                    latency_list = [item['current_frame'] - item['detection_id'] for item in data]
                    dead_frames = [x["result"] for x in data].count(None)
                    average_latency = sum(latency_list) / len(latency_list)
                    average_iou = sum(iou_list) / len(iou_list)


                    with open(filename, 'w') as csvfile:
                        # creating a csv writer object
                        csvwriter = csv.writer(csvfile)

                        # writing the fields
                        csvwriter.writerow(fields)

                        # writing the data rows
                        csvwriter.writerows(rows)

                    print(f'Average IOU, deadframes, latency is {average_iou, dead_frames, average_latency}')
                exit()
            except UnpicklingError:
                print('Hmmm something is wrong')
        else:
            print('This script is not written to recognize this file')
            exit()

