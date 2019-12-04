import os
import argparse
import numpy as np
import cv2
from detector import Detector
from tracker import Tracker

parser = argparse.ArgumentParser()

parser.add_argument("--video_source", default=r"C:\Users\Thanh\Downloads\vinhphuc_cam02.stream_2019-11-03-06.42.05.890.mp4", type=str)
parser.add_argument("--object_detector_model_dir", default=r"E:\PythonProjects\MeatDeli_Pedestrian_Detection\Object_Detection_Inference\ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03", type=str)
parser.add_argument("--path_to_reid_model", default=r"E:\PythonProjects\MeatDeli_Pedestrian_Detection\tracking_without_bells_and_whistles\mars-small128.pb", type=str)
parser.add_argument("--detection_person_thresh", default=0.4, type=float)
parser.add_argument("--inactive_steps_before_removed", default=1000, type=int)
parser.add_argument("--max_features_num", default=10, type=int)
parser.add_argument("--reid_sim_threshold", default=1.0, type=float)
parser.add_argument("--reid_iou_threshold", default=0.6, type=float)
parser.add_argument("--max_traject_steps", default=25, type=int)

args = parser.parse_args()

video_source = args.video_source
object_detector_model_dir = args.object_detector_model_dir
path_to_reid_model = args.path_to_reid_model
detection_person_thresh = args.detection_person_thresh
inactive_steps_before_removed = args.inactive_steps_before_removed
max_features_num = args.max_features_num
reid_sim_threshold = args.reid_sim_threshold
reid_iou_threshold = args.reid_iou_threshold
max_traject_steps = args.max_traject_steps

detector = Detector(model_dir=object_detector_model_dir,
                    path_to_reid_model=path_to_reid_model,
                    detection_person_thresh=detection_person_thresh)

tracker = Tracker(inactive_steps_before_removed=inactive_steps_before_removed,
                  max_features_num=max_features_num,
                  reid_sim_threshold=reid_sim_threshold,
                  reid_iou_threshold=reid_iou_threshold,
                  max_traject_steps=max_traject_steps)

cap = cv2.VideoCapture(video_source)

while True:
    ret, img = cap.read()
    if not ret:
        break

    detections = detector(frame=img)
    tracker.step(detections)

    results = tracker.get_result()

    for result in results:
        bbox = result[0]
        id = result[1]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)
        cv2.putText(img, "Id:" + str(id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(255, 255, 255), thickness=1)

    #for detection in detections:
    #    bbox = detection.bbox
    #    #print(bbox)
    #    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

    cv2.imshow("", img)
    cv2.waitKey(1)
