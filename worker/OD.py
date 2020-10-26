import cv2
import numpy as np

class Tensorflow:
    def __init__(self):
        print('[INFO] Processing tensorflow as detector...')
        import os
        import sys
        import tarfile
        import tensorflow as tf
        from glob import glob
        from os.path import dirname, join, basename

        sys.path.append("..")

        import utils.label_map_util

        physical_devices = tf.config.experimental.list_physical_devices('GPU')

        try:
            tf.config.experimental.set_virtual_device_configuration(
                physical_devices[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        models = glob(join(dirname(__file__), "*.tar.gz"))
        models = [model.split('\\')[-1].strip('.tar.gz') for model in models]

        for i, model in enumerate(models):
            print(f'{i}: {model}')

        while True:
            try:
                model_name = models[int(input('Enter your model index: '))]
                break
            except:
                print('Wrong Input Format')

        model_file = model_name + '.tar.gz'

        path_to_ckpt = model_name + '/frozen_inference_graph.pb'

        path_to_labels = os.path.join('data', 'mscoco_label_map.pbtxt')

        num_classes = 90

        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        label_map = utils.label_map_util.load_labelmap(path_to_labels)
        categories = utils.label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                          use_display_name=True)
        self.category_index = utils.label_map_util.create_category_index(categories)
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

    def detection(self, frame, msg):

        image_np_expanded = np.expand_dims(frame, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        boxes, scores, classes, num_detections = self.sess.run(
            [boxes,
             scores,
             classes,
             num_detections
             ],
            feed_dict={image_tensor: image_np_expanded}
        )
        frame_ = self.generate_valid_info(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index
        )

        return frame_

    @staticmethod
    def generate_valid_info(image_,
                            boxes_, classes_,
                            scores_, category_index_,
                            max_boxes_to_draw=20, min_score_thresh=.5
                            ):
        """
        this is to filter valid detection of assurance of over 50%
         and then draw on it
        :param image_:image
        :param boxes_:
        :param classes_:
        :param scores_:
        :param category_index_:
        :param max_boxes_to_draw:
        :param min_score_thresh:
        :return: drawn image
        """
        detected_objects = []
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes_.shape[0]

        for i in range(min(max_boxes_to_draw, boxes_.shape[0])):
            if scores_ is None or scores_[i] > min_score_thresh:
                box = tuple(boxes_[i].tolist())
                detected_objects.append(
                    [
                        box,
                        scores_[i],
                        category_index_[classes_[i]]['name']
                    ]
                )

        return detected_objects


class CV2:
    def __init__(self):
        print('[INFO] Processing cv2 as detector...')
        self.net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel'
        )

    @staticmethod
    def detection(frame):
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height),
                                     (104.0, 177.0, 123.0)
                                     )
        net.setInput(blob)
        detections = net.forward()
        rectangles = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            object_ = detections[0, 0, i, 2]
            if object_ > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                rectangles.append([box.astype("int"), object_])

        return rectangles
