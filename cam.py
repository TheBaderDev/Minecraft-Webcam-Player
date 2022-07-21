import cv2
import mediapipe as mp
import numpy as np
import uuid
import os
import pyautogui
from shapely.geometry import Point, LineString
import traceback
import math
import pynput
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

state_data = {
    "face_direction_lr": 90,
    "face_direction_ud": 0,
    "is_crouching": False,
    "jumping": False,
    "place_item": False,
    "break_item": False,
    "in_inventory": False
}

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            body_results = pose.process(image)
            hand_results = hands.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Check for body landmarks
            try:
                landmarks = body_results.pose_landmarks.landmark
                # angle = calculate_angle(shoulder, elbow, wrist)

                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                
                # #Checks where the user's face is aiming and moves the mouse left and right
                shoulder_midpoint = [(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x+landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x)/2,(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y+landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)/2]

                #TURNING MOUSE LEFT AND RIGHT
                right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                right_d = math.dist(nose, right_ear)
                left_d = math.dist(nose, left_ear)
                #Convert this info into an angle between 0 and 180 degrees
                #0 is right, 180 is left
                angle = 90
                if(right_d > left_d):
                    angle = (right_d/(right_d+left_d))*180
                elif(left_d > right_d):
                    angle = 180-(left_d/(right_d+left_d))*180
                # print(str(angle)+" | right: "+str(right_d)+" | left: "+str(left_d))
                
                #a normal face turning angle range seems to be between 40 and 140
                print(angle)
                max_turn_angle = 40
                max_turn_speed = 50
                if angle < 90-max_turn_angle:
                    pyautogui.moveRel(max_turn_speed,0)
                elif angle > 90+max_turn_angle:
                    pyautogui.moveRel(-1*max_turn_speed,0)
                else:
                    screen_width, screen_height = pyautogui.size()
                    m_x, m_y = pyautogui.position()
                    pyautogui.moveTo(screen_width*(abs(180-angle)/180),m_y)


                # if angle < 90: #looking right move mouse left
                #     mouse.move(angle,0)
                # elif angle > 90: #looking left move mouse right
                #     mouse.move(angle,0)
                # sensitiv = 1



            except Exception:
                pass
                traceback.print_exc()
            
            # Render detections
            mp_drawing.draw_landmarks(image, body_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )       
            if hand_results.multi_hand_landmarks:
                for num, hand in enumerate(hand_results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )
                    
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()