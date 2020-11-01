from tensorflow.compat.v1 import ConfigProto, InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
InteractiveSession(config = config)

import cv2
from mtcnn.mtcnn import MTCNN
import sys

# function to outline the faces in the image
def outline_faces(img):

    detector = MTCNN() # loading the detector
    faces = detector.detect_faces(img) # detect faces in the image

    for face in faces:

        # getting the coordinates of the bounding box
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3) # outlining the faces 
        
    return img

cap = cv2.VideoCapture(0) # trying to capture video from the camera

# exiting if the camera is not accessible
if not cap.isOpened():
    sys.exit("Cannot open camera!")

while True:
    ret, frame = cap.read()

    # exiting if images are not received from the camera
    if not ret:
        sys.exit("Cannot receive frames. Exiting...")

    frame = cv2.resize(frame, (640, 480)) # resizing the images to (640, 480)
    frame = cv2.flip(frame, 1)

    frame = outline_faces(frame) # outlining the faces in the image

    cv2.imshow('frame', frame) # displaying the frames
    if cv2.waitKey(200) == ord("q"): # exiting the program if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()

    