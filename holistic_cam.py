import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import pyautogui
import traceback
import math
import time
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

x_offset = 70  #bc the face tracker lr is limited to 35-145 degrees (35 on each end)
y_offset = 140 #mid 85, low 60 - high 105

# returns the left and right angle and up and down angle (0-180)
def getFaceDirection(results):
    pose_landmarks = results.pose_landmarks.landmark
    face_landmarks = results.face_landmarks.landmark

    # angle = calculate_angle(shoulder, elbow, wrist)

    nose = [pose_landmarks[mp_pose.PoseLandmark.NOSE.value].x,pose_landmarks[mp_pose.PoseLandmark.NOSE.value].y]
    l_shoulder = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    r_shoulder = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    
    #TURNING MOUSE LEFT AND RIGHT
    right_ear = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,pose_landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
    left_ear = [pose_landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,pose_landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    right_d = math.dist(nose, right_ear)
    left_d = math.dist(nose, left_ear)
    #Convert this info into an angle between 0 and 180 degrees (0 is right, 180 is left)
    angleX = 90
    if(right_d > left_d):
        angleX = (right_d/(right_d+left_d))*180
    elif(left_d > right_d):
        angleX = 180-(left_d/(right_d+left_d))*180
    angleX = angleX - x_offset/2

    #MOVING MOUSE UP AND DOWN
    angleY = 90
    shoulder_midpoint = [(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x+pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)/2,(pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y+pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)/2]
    mesh_nose = [face_landmarks[94].x, face_landmarks[94].y]
    chin = [face_landmarks[175].x, face_landmarks[175].y]
    forehead = [face_landmarks[151].x, face_landmarks[151].y]
    lower_dist = math.dist(mesh_nose, chin)
    upper_dist = math.dist(mesh_nose, forehead)
    if(lower_dist > upper_dist):
        angleY = (lower_dist/(lower_dist+upper_dist))*180
    elif(upper_dist > lower_dist):
        angleY = 180-(upper_dist/(lower_dist+upper_dist))*180
    # print(str(math.dist(mesh_nose, chin))+" | "+str(math.dist(mesh_nose, forehead)))
    angleY = angleY - y_offset/2
    return (angleX,angleY)

def moveCamera(angleX, angleY):
    screen_width, screen_height = pyautogui.size()
    m_x, m_y = pyautogui.position()

    #correcting for offset and direction
    x_range = 180-x_offset
    y_range = 180-y_offset
    pyautogui.moveTo(screen_width*(abs(x_range-angleX)/x_range),screen_height*(abs(y_range-angleY)/y_range), duration=0.3)
    # pyautogui.moveTo(screen_width*(angleX),screen_height*(angleY), duration=0.1)
    return


cap = cv2.VideoCapture(0)
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # print(getFaceDirection(results)
        try:
            angleX, angleY = getFaceDirection(results)
            moveCamera(angleX, angleY)
        except Exception:
            # traceback.print_exc()
            pass
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
                        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()