import numpy as np
import torch
import tensorflow as tf
from ultralytics import YOLO

# =======================================================================================================================

# tf.config.list_physical_devices('GPU')
# print()
#
# print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.mps.is_built()}")
# print(f"MPS 장치가 사용 가능한가? {torch.backends.mps.is_available()}")
# device = torch.device("mps")
#
#
# # =======================================================================================================================
# # Load a model
# model = YOLO('/Users/bagsangbeom/PycharmProjects/DMS_project/weights/best (12월 16일).pt')  # build a new model from YAML
# detector = model.predict(source=0, show=True, conf=0.4, save=True)



# # ====================================================================================================================
# import supervision as sv
# from ultralytics import YOLO
#
# model = YOLO('/Users/bagsangbeom/PycharmProjects/DMS/weights/best (12월 16일).pt')
# byte_tracker = sv.ByteTrack()
# annotator = sv.BoxAnnotator()
#
# def callback(frame: np.ndarray, index: int) -> np.ndarray
# 	results = model(frame)[0]
# 	detections = sv.Detections.from_ultralytics(results)
# 	detections = byte_tracker.update_with_detections(detections)
# 	labels = [
# 		f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
# 		for _,_, confidence, class_id, tracker_id
# 		in detections
# 	]
# 	return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
#
# sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)

# # ====================================================================================================================
import cv2
import argparse

from ultralytics import YOLO
import supervision as sv

def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="YOLOv8 live")
	parser.add_argument(
		"--webcam-resolution",
		default=[1280,720],
		nargs=2,
		type=int
	)
	args = parser.parse_args()
	return args
	
	
def main():
	args = parse_arguments()
	frame_width, frame_height = args.webcam_resolution
	
	cap = cv2.VideoCapture(1)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
	
	model = YOLO('/Users/bagsangbeom/PycharmProjects/DMS/weights/best (12월 16일).pt')
	
	box_annotator = sv.BoxAnnotator(
		thickness=2,
		text_thickness=2,
		text_scale=1
	)
	
	while True:
		ret, frame = cap.read()
		
		result = model(frame)[0]
		detections = sv.Detections.from_yolov8(result)
		labels = [
			f"{model.model.names[class_id]} {confidence:0.2f}"
			for _, confidence, calss_id,
			in detections
		]
		frame = box_annotator.annotate(
			scene=frame,
			detections=detections,
			labels=labels
		)
		
		cv2.imshow("yolov8", frame)

		if (cv2.waitKey(30) == 27):
			break
		
if __name__ == "__main__":
	main()
