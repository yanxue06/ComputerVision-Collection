import cv2
import numpy as np 

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the defauly webCam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

i = 0 

while True:
    success, frame = cap.read()  # Read video frame (a NumPy Array)

    if not success:
        print("Error: Failed to capture image.")
        break

    if i == 0:
        print(frame.shape) # print the shape once only 
        i += 1

    frame = cv2.flip(frame, 1)  # Flip horizontally for selfie mode
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # grey scale way faster for rendering 
    # scale factor is how the computer shrinks the image on each render, such that the size of the face grows relative to the rest of the image 
    # minNeighbours is the amount of rectangles needed to be detected in order for a rectangle to be rendered as a face
    # minSize means a face needs to be at least 50 by 50 

    # Draw a rectangle around detected faces
    if i == 1:
        print(frame.shape) # print the shape once only again 
        i += 1
        
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        # 1st param - the image to draw on 
        # 2nd param - Top left corner of rectangle (about the face)
        # 3rd param - bottome right corner of the rectangel (makes sense that x at which the face starts + the width of the face)
        # note for 3rd param - moving down increases y which is why it is y + h 
        # 4th param - for color scheme in rgb, that is the code for green 
        # 5th param - the thickness of the box 

    # Show video with face tracking !! 
    cv2.imshow("Selfie Mode Face Tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
