import copy
import time
from collections import deque
import numpy as np


class Track(object):
    def __init__(self, bbox, score, track_id, feature, inactive_steps_before_removed, max_feature_num, max_traject_steps):
        self.id = track_id
        self.bbox = bbox
        self.score = score
        self.features = deque([copy.deepcopy(feature)], maxlen=max_feature_num)
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.inactive_steps = 0
        self.traject_pos = deque([copy.deepcopy(bbox)], maxlen=max_traject_steps)
        self.traject_vel = deque([], maxlen=max_traject_steps)
        self.time_stamp = deque([time.time()], maxlen=max_traject_steps)
        self.birth_time = [time.time()]
        self.alive_time = []

    def add_feature(self, feature):
        self.features.append(copy.deepcopy(feature))

    def compute_distance(self, test_feature):
        if len(self.features) > 1:
            features = np.stack(self.features, axis=0)
        else:
            features = self.features[0]

        features = np.mean(features, axis=0, keepdims=False)

        dist = np.linalg.norm(features - test_feature, 2)

        return dist

    def reset_trajectory(self):
        self.traject_pos.clear()
        self.traject_vel.clear()
        self.traject_pos.append(copy.deepcopy(self.bbox))
