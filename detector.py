import os
from distutils.version import StrictVersion
import numpy as np
import cv2
import tensorflow as tf
from detection import Detection
from reid_network import create_box_encoder
from utils import to_ltrb

if StrictVersion(tf.__version__) < StrictVersion("1.9.0"):
    raise ImportError("Please upgrade your TensorFlow installation to v1.9.* or later!")


class Detector(object):
    def __init__(self, model_dir, path_to_reid_model, detection_person_thresh=0.4):
        self.path_to_frozen_graph = os.path.join(model_dir, "frozen_inference_graph.pb")
        self.reid_model = create_box_encoder(model_filename=path_to_reid_model)
        self.detection_person_thresh = detection_person_thresh
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_frozen_graph, "rb") as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")

        self.sess = tf.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name("image_tensor:0")
        self.boxes = self.detection_graph.get_tensor_by_name("detection_boxes:0")
        self.scores = self.detection_graph.get_tensor_by_name("detection_scores:0")
        self.classes = self.detection_graph.get_tensor_by_name("detection_classes:0")
        self.num_detections = self.detection_graph.get_tensor_by_name("num_detections:0")

    def __call__(self, frame):
        height, width = frame.shape[:2]
        #print(height, width)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bgr_expanded = np.expand_dims(frame_bgr, 0)

        (run_boxes, run_scores, run_classes, run_num_detections) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                                                                 feed_dict={self.image_tensor: frame_bgr_expanded})

        run_boxes = np.squeeze(run_boxes)
        run_scores = np.squeeze(run_scores)
        run_classes = np.squeeze(run_classes)
        run_num_detections = np.squeeze(run_num_detections)

        indices = np.intersect1d(np.where(run_scores > self.detection_person_thresh)[0], np.where(run_classes == 1.)[0])
        boxes = []
        scores = []
        for box, score in zip(run_boxes[indices], run_scores[indices]):
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            boxes.append([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1])
            scores.append(score)
        #print(boxes)
        features = list(self.reid_model(frame_bgr, boxes))
        detections = []
        for box, score, feature in zip(boxes, scores, features):
            detections.append(Detection(to_ltrb(box), score, feature))
            #print(to_ltrb(box))

        return detections
