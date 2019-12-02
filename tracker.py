import os
from itertools import product
import copy
import time
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from detection import Detection
from track import Track
from reid_network import create_box_encoder
from utils import bb_intersection_over_union, iou_mat, non_maximum_suppression


class Tracker(object):

    def __init__(self, inactive_steps_before_removed, max_features_num, reid_sim_threshold, reid_iou_threshold, max_traject_steps):
        #self.reid_model = create_box_encoder(model_filename=path_to_reid_model)
        #self.detection_person_thresh = detection_person_thresh
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.max_features_num = max_features_num
        self.reid_sim_threshold = reid_sim_threshold
        self.reid_iou_threshold = reid_iou_threshold
        self.max_traject_steps = max_traject_steps

        self.active_tracks = []
        self.inactive_tracks = []
        self.track_num = 0

    def tracks_to_inactive(self, tracks):
        self.active_tracks = [t for t in self.active_tracks if t not in tracks]
        for t in tracks:
            t.bbox = t.traject_pos[-1]
            t.alive_time.append(time.time() - t.birth_time[-1])
        self.inactive_tracks += tracks

    def add_new_tracks(self, detections_list):
        num_new = len(detections_list)
        for i, detection in enumerate(detections_list):
            self.active_tracks.append(Track(bbox=detection.bbox, score=detection.score, track_id=self.track_num + i,
                                            feature=detection.feature,
                                            inactive_steps_before_removed=self.inactive_steps_before_removed,
                                            max_feature_num=self.max_features_num,
                                            max_traject_steps=self.max_traject_steps))

        self.track_num += num_new

    def get_active_bboxes(self):
        if len(self.active_tracks) == 1:
            bboxes = self.active_tracks[0].bbox
        elif len(self.active_tracks) > 1:
            bboxes = [t.bbox for t in self.active_tracks]
        else:
            bboxes = []

        return bboxes

    def get_active_scores(self):
        if len(self.active_tracks) == 1:
            scores = self.active_tracks[0].score
        elif len(self.active_tracks) > 1:
            scores = [t.score for t in self.active_tracks]
        else:
            scores = []

        return scores

    def add_features(self, new_features):
        for t, f in zip(self.active_tracks, new_features):
            t.add_feature(f)

    def match_reid_iou_sim(self, detections):

        def distance_feature_between_track_and_detection(track, detection):
            dist = track.compute_distance(detection.feature)
            return dist

        active_tracks_bboxes = self.get_active_bboxes()
        detection_bboxes = [detection.bbox for detection in detections]
        detection_features = [detection.feature for detection in detections]

        iou_matrix = iou_mat(active_tracks_bboxes, detection_bboxes)

        dist_matrix = np.asarray(list(map(lambda x: distance_feature_between_track_and_detection(track=x[0], detection=x[1]), product(self.active_tracks, detections))))

        iou_mask = np.greater_equal(iou_matrix, self.reid_iou_threshold)

        neg_iou_mask = ~iou_mask

        dist_matrix = dist_matrix * iou_mask.astype(np.float) + dist_matrix * neg_iou_mask.astype(np.float) * 1000

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_detection_indx = []
        matched_track_indx = []

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= self.reid_sim_threshold:
                matched_track_indx.append(r)
                matched_detection_indx.append(c)
                t = self.active_tracks[r]
                t.bbox = detections[c].bbox
                t.score = detections[c].score
                t.add_feature(detections[c].feature)
        unmatched_tracks = [self.active_tracks[i] for i in range(len(self.active_tracks)) if i not in matched_track_indx]
        unmatched_detections = [detections[j] for j in range(len(detections)) if j not in matched_detection_indx]

        self.active_tracks = [self.active_tracks[k] for k in matched_track_indx]

        #self.tracks_to_inactive(unmatched_tracks)

        return unmatched_tracks, unmatched_detections

    def match_reid_sim(self, detections):

        def distance_feature_between_track_and_detection(track, detection):
            dist = track.compute_distance(detection.feature)
            return dist

        dist_matrix = np.asarray(list(map(lambda x: distance_feature_between_track_and_detection(track=x[0], detection=x[1]), product(self.inactive_tracks, detections))))

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        matched_detection_indx = []
        matched_track_indx = []

        for r, c in zip(row_ind, col_ind):
            if dist_matrix[r, c] <= self.reid_sim_threshold:
                matched_track_indx.append(r)
                matched_detection_indx.append(c)
                t = self.inactive_tracks[r]
                t.reset_trajectory()
                t.bbox = detections[c].bbox
                t.score = detections[c].score
                t.inactive_steps = 0
                t.add_feature(detections[c].feature)
                t.birth_time.append(time.time())
                self.active_tracks.append(t)
        matched_tracks = [self.inactive_tracks[i] for i in range(len(self.inactive_tracks)) if i in matched_track_indx]
        unmatched_detections = [detections[j] for j in range(len(detections)) if j not in matched_detection_indx]

        for t in matched_tracks:
            self.inactive_tracks.remove(t)

        #return matched_tracks, unmatched_detections
        return matched_tracks, unmatched_detections

    def motion_step(self, track):
        track.bbox = track.bbox + track.traject_vel[-1] * (time.time() - track.time_stamp[-1]) * 1000

    def motion(self):
        for t in self.active_tracks:
            last_bbox = t.traject_pos
            moments = t.time_stamp

            vs = np.asarray([1000 * (p2 - p1) / (t2 - t1) for p1, p2, t1, t2 in zip(last_bbox, last_bbox[1:], moments, moments[1:])], dtype=np.float)
            vs = np.mean(vs)

            t.traject_vel.append(vs)

            self.motion_step(t)

    def step(self, detections):

        for t in self.active_tracks:
            t.traject_pos.append(copy.deepcopy(t.bbox))

        self.motion()

        if len(self.active_tracks) > 0 and len(detections) > 0:
            unmatched_active_tracks, unmatched_detections = self.match_reid_iou_sim(detections)

            if len(unmatched_detections) > 0:
                if len(self.inactive_tracks) > 0:
                    unmatched_detections = self.match_reid_sim(unmatched_detections)
                else:
                    pass
            else:
                pass

            if len(unmatched_active_tracks) > 0:
                self.tracks_to_inactive(unmatched_active_tracks)
            else:
                pass

            if len(unmatched_detections) > 0:
                self.add_new_tracks(unmatched_detections)
            else:
                pass

        elif len(self.active_tracks) > 0 and len(detections) == 0:
            self.tracks_to_inactive(self.active_tracks)

        elif len(self.active_tracks) == 0 and len(detections) > 0:
            if len(self.inactive_tracks) > 0:
                unmatched_detections = self.match_reid_sim(detections)
                if len(unmatched_detections) > 0:
                    self.add_new_tracks(unmatched_detections)
                else:
                    pass
            else:
                self.add_new_tracks(detections)

        else:
            pass

        remove_inactive = []
        for t in self.inactive_tracks:
            t.inactive_steps += 1
            if t.inactive_steps > t.inactive_steps_before_removed:
                remove_inactive.append(t)

        #self.track_num -= len(remove_inactive)
        for track in remove_inactive:
            self.inactive_tracks.remove(track)

    def get_result(self):
        results = []

        for t in self.active_tracks:
            results.append((t.bbox, t.id, t.score))

        return results
