import argparse
import numpy as np
import cv2
from detector import Detector

parser = argparse.ArgumentParser()

parser.add_argument("--video_source", default=r"C:\Users\Thanh\Downloads\vinhphuc_cam02.stream_2019-11-03-06.42.05.890.mp4", type=str)
parser.add_argument("--object_detector_model_dir", default=r"E:\PythonProjects\MeatDeli_Pedestrian_Detection\Object_Detection_Inference\ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03", type=str)
parser.add_argument("--path_to_reid_model", default=r"E:\PythonProjects\MeatDeli_Pedestrian_Detection\tracking_without_bells_and_whistles\mars-small128.pb", type=str)

args = parser.parse_args()

video_source = args.video_source
object_detector_model_dir = args.object_detector_model_dir
path_to_reid_model = args.path_to_reid_model

detector = Detector(model_dir=object_detector_model_dir, path_to_reid_model=path_to_reid_model)

cap = cv2.VideoCapture(video_source)

while True:
    ret, img = cap.read()

    if not ret:
        break

    detections = detector(frame=img)
    print(len(detections))
    for detection in detections:
        print(detection.bbox, detection.score, detection.feature.shape)
        bbox = detection.bbox
        score = detection.score
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        cv2.putText(img, str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

    cv2.imshow("", img)
    cv2.waitKey(1)
