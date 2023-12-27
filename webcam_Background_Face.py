import cv2
from src.face_mesh_Utils import ExorcistFace

show_webcam = True
max_people = 1

# Image to swap face with
exorcist_image_url = "https://media.sketchfab.com/models/19a3001d053948d982126692396115c3/thumbnails/61850e1a2745428280a320f41a18cff3/1d09a93634ac4e339ba0a38912f2c20f.jpeg"
# Initialize ExorcistFace class
draw_exorcist = ExorcistFace(exorcist_image_url, show_webcam, max_people)

# Initialize webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Exorcist face", cv2.WINDOW_NORMAL)

while cap.isOpened():
	
	# Read frame
	ret, frame = cap.read()
	
	if not ret:
		continue
	
	# Flip the image horizontally
	frame = cv2.flip(frame, 1)
	
	ret, exorcist_image = draw_exorcist(frame)
	
	if not ret:
		continue
	
	cv2.imshow("Exorcist face", exorcist_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break