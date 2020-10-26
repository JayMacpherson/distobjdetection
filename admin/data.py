import numpy as np
import pickle


def serialize(data, entry: int = 0):
    """
    this convert data to list or bytes
    :param data: can be list or bytes
    :param entry: 1 if you intend to convert list
        to bytes but 0 to convert bytes to list
    :return: bytes or list
    """
    try:
        if entry:
            return pickle.dumps(data)
        else:
            return pickle.loads(data)
    except:
        return False


def extract(data):
    """
    :param data: bytes data
    :return: message and image
    """
    raw = np.frombuffer(
        data[3],
        dtype=data[1]
    )
    return data[0], raw.reshape(data[2])


def prepare(frame, text: str):
    """
    :param frame: opencv frame read
    :param text: text you intend to merge with the frame
    :return: byte data of the frame, text and frame info
    """
    if not frame.flags['C_CONTIGUOUS']:
        frame = np.ascontiguousarray(frame)
    data = [
        text,
        str(frame.dtype),
        frame.shape,
        frame,
        False
    ]
    return serialize(data, entry=1)