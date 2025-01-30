import cv2
import numpy as np 
import mediapipe as mp 

# Used to convert protobuf message to a dictionary. 
from google.protobuf.json_format import MessageToDict 

# Initializing the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the defauly webCam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


while True:
    success, frame = cap.read()  # Read video frame (a NumPy Array)

    if not success:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for selfie mode

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale, reqd for OpenCV Face
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB, reqd for MediaPipe

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    # checking if hands are present
    results = hands.process(imgRGB)  

    # grey scale way faster for rendering 
    # scale factor is how the computer shrinks the image on each render, such that the size of the face grows relative to the rest of the image 
    # minNeighbours is the amount of rectangles needed to be detected in order for a rectangle to be rendered as a face
    # minSize means a face needs to be at least 50 by 50 
        
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        # 1st param - the image to draw on 
        # 2nd param - Top left corner of rectangle (about the face)
        # 3rd param - bottome right corner of the rectangel (makes sense that x at which the face starts + the width of the face)
        # note for 3rd param - moving down increases y which is why it is y + h 
        # 4th param - for color scheme in rgb, that is the code for green 
        # 5th param - the thickness of the box 

    if results.multi_hand_landmarks: 
  
        # Both Hands are present in image(frame) 
        if len(results.multi_handedness) == 2: 
                # Display 'Both Hands' on the image 
            cv2.putText(frame, 'Both Hands', (250, 50), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.9, (0, 255, 0), 2) 
  
        # If any hand present 
        else: 
            for i in results.multi_handedness: 
                
                # Return whether it is Right or Left Hand 
                hand_dict = MessageToDict(i) 

                label = hand_dict['classification'][0]['label']
  
                if label == 'Left': 
                    
                    # Display 'Left Hand' on 
                    # left side of window 
                    cv2.putText(frame, label+' Hand', 
                                (20, 50), 
                                cv2.FONT_HERSHEY_COMPLEX,  
                                0.9, (0, 255, 0), 2) 
  
                if label == 'Right': 
                      
                    # Display 'Left Hand' 
                    # on left side of window 
                    cv2.putText(frame, label+' Hand', (460, 50), 
                                cv2.FONT_HERSHEY_COMPLEX, 
                                0.9, (0, 255, 0), 2) 

    # Show video with face tracking !! 
    cv2.imshow("face and hand tracking", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break