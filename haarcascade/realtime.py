import cv2
import sys

def outline_faces(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # conveting image to grayscale
    
    frontalface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # classifier for frontal face detection
    profileface = cv2.CascadeClassifier('haarcascade_profileface.xml') # classifier for profile face detection

    frontal = frontalface.detectMultiScale(gray, 1.2, 4) # detect frontal faces
    profile = profileface.detectMultiScale(gray, 1.2, 4) # detect profile faces

    # outline frontal faces, if any
    if len(frontal) != 0:
        for face in frontal:
            (x, y, w, h) = face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    
    # outline profile faces, if any
    if len(profile) != 0:
        for face in profile:
            (x, y, w, h) = face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    return img

cap = cv2.VideoCapture(0) # trying to capture video from the camera

# exiting if the camera is not available
if not cap.isOpened():
    sys.exit("Cannot open camera!")

while True:
    ret, frame = cap.read()

    # exiting if images are not received from the camera
    if not ret:
        sys.exit("Cannot receive frames. Exiting...")

    frame = cv2.resize(frame, (640, 480)) # resizing the images to (640, 480)
    frame = cv2.flip(frame, 1) # flipping the image vertically

    frame = outline_faces(frame) # outlining the faces in camera

    cv2.imshow('frame', frame) # displaying the frames
    if cv2.waitKey(1) == ord('q'): # exiting the program if 'q' is pressed
        break

cap.release()
cv2.destroyAllWindows()

