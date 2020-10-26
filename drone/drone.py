import cv2
import zmq
from time import sleep
from data import extract, prepare
from PIL import Image

# captures the file or webcam
cap = cv2.VideoCapture(0)

# connects to the edge server
socket = zmq.Context()
socket = socket.socket(zmq.REQ)

socket.connect("tcp://localhost:4044")
print('[INFO] Stream as started!')

while True:
    # reads for frame and prepares it
    ret, frame = cap.read()
    data = prepare(frame, 'Sender')

    # sends prepared data
    socket.send(data)

    # receives feedback from edge server
    data = socket.recv()

    # displays feedback frame on opencv window
    if data != b'None':
        msg, image = extract(data)
        cv2.imshow(msg, image)
        cv2.waitKey(1)
