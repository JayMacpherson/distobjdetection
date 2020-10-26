import zmq
import cv2
import pickle
from multiprocessing import Process, Queue
from data import extract, prepare


UI = True


# base class to store queue class in memory
class Base(Process):
    """
    inherit Process and holds address
    """

    def __init__(self, queue=None):
        super().__init__()
        self.queue = queue


class Networking(Base):

    def run(self):
        # connects to edge server
        print('[INFO] Connecting to Edge Node Server...')

        # connecting to server
        socket = zmq.Context()
        socket = socket.socket(zmq.REQ)
        socket.connect("tcp://localhost:4022")

        print('[INFO] Admin is in active state!')

        while True:

            socket.send(b'None')

            # to receive server data response
            data = socket.recv()

            if data != b'None':

                # extracting the server sent packet
                data = pickle.loads(data)

                # extracting the message and frame from the extracted packet seg
                raw_frame = data.get('raw_frame', None)
                ui_data = {}

                if raw_frame:
                    msg, frame = extract(raw_frame)

                    # getting the height and width for further computation
                    height, width = frame.shape[:2]

                    no = 0
                    _accuracy = 0

                    for box in data['detection_data']:
                        box, accuracy, name = box

                        _accuracy += accuracy
                        no += 1

                        # calculating the object position and labeling it
                        ymin, xmin, ymax, xmax = box
                        try:
                            h = int((ymax - ymin) * height)
                            w = int((xmax - xmin) * width)
                            y = int((height * ymax) - h)
                            x = int((width * xmax) - w)

                            # labeling object
                            frame = cv2.putText(
                                frame, f'{name} {str(accuracy * 100)[:2]}%',
                                (x - 5, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (225, 0, 0), 2
                            )

                            # drawing a box around the detected object
                            frame = cv2.rectangle(
                                frame,
                                (x, y),
                                (x + w, y + h),
                                (255, 0, 0), 2
                            )
                        except:
                            pass

                    # calculating the mean accuracy of the detected objects
                    try:
                        _accuracy = _accuracy/no
                    except:
                        pass
                    data['worker']['accuracy'] = _accuracy
                    ui_data = data['worker']
                    # shows streams
                    cv2.imshow(msg, frame)
                    cv2.waitKey(1)
                else:
                    ui_data = data

                if ui_data:
                    if UI:
                        # sends statistics to ui if ui is activated
                        self.queue.put(ui_data)
                    else:
                        print(ui_data)


if __name__ == '__main__':
    if UI:
        print('[INFO] Launching UI...')
        from kivy.config import Config

        Config.set('graphics', 'width', '900')
        Config.set('graphics', 'height', '500')
        from kivy.uix.floatlayout import FloatLayout
        from kivy.app import App
        from kivy.factory import Factory
        from kivy.uix.button import Button
        from kivymd.app import MDApp
        from kivy.clock import Clock
        from kivy.core.window import Window

        from random import randint

        class StatsLayout(FloatLayout):
            background = f'bg/{randint(2, 7)}.jpg'

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                Window.bind(on_request_close=self.on_request_close)
                Clock.schedule_interval(self.update_stat, .1)

                self.queue = Queue()
                self._net = Networking(queue=self.queue)
                self._net.start()

                self.im = 2
                self.workers = []
                self.dropped = 0
                self.mean_total_fps = [0, 0]

            def update_stat(self, _):
                if not self.queue.empty():
                    data = self.queue.get()
                    self.queue.empty()

                    worker_name = data.get('worker_name')
                    if worker_name:
                        if worker_name not in self.workers:
                            new = {
                                'name': data['worker_name'],
                                'fps': self.accuracy(data['worker_speed'], False),
                                'accuracy': self.accuracy(data.get('accuracy'), True),
                                'lossy': str(data['lossy_encoding']),
                                'status': data['status']
                            }

                            self.ids.workers.data.append(new)
                            self.workers.append(worker_name)
                        else:
                            _index = self.workers.index(worker_name)

                            self.ids.workers.data[_index] = {
                                'name': worker_name,
                                'fps': self.accuracy(data['worker_speed'], False),
                                'accuracy': self.accuracy(data['accuracy'], True),
                                'lossy': str(data['lossy_encoding']),
                                'status': data['status']
                            }

                    # to refresh the worker list values
                    self.ids.workers.refresh_from_data()

                    if self.mean_total_fps[1] > 10:
                        mean = self.mean_total_fps[0]/self.mean_total_fps[1]
                        self.ids.Fps.text = str(1/mean)[:4]
                        self.mean_total_fps = [0, 0]
                    else:
                        self.mean_total_fps[0] += data['worker_speed']
                        self.mean_total_fps[1] += 1

                    self.dropped += data['dropped_frames']
                    self.ids.Drop.text = str(self.dropped)
                    self.ids.Accuracy.text = self.accuracy(data['accuracy'], True)

            @staticmethod
            def accuracy(value, entry):
                if not value:
                    return str(0)
                else:
                    if entry:
                        return f'{str(value * 100)[:2]}%'
                    else:
                        return str(1/value)[:4]

            def on_request_close(self, *args):
                self._net.terminate()

        class Stats(MDApp):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.title = 'Distributive Stats'
                self.theme_cls.primary_palette = 'Teal'

            def build(self):
                # noinspection PyAttributeOutsideInit
                self.root = Factory.StatsLayout()

        Stats().run()

    else:
        net = Networking()
        net.start()

        net.join()
