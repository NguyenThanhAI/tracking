import numpy as np


class Detection(object):
    def __init__(self, bbox, score, feature):
        self.bbox = bbox
        self.score = score
        self.feature = feature

    def to_ltwh(self):
        ret = self.bbox.copy()
        ret[2:] -= ret[:2]

        return ret
