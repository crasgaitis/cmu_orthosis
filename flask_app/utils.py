import math
import cv2
import mediapipe as mp
import numpy as np

def detect_finger(frame, hands, mp_hands, mp_drawing):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            h, w, _ = frame.shape
            tip_x, tip_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            mcp_x, mcp_y = int(index_finger_mcp.x * w), int(index_finger_mcp.y * h)

            # len of the finger  = distance between tip & base
            finger_length = (np.sqrt((tip_x - mcp_x) ** 2 + (tip_y - mcp_y) ** 2)) / 1.8
            finger_thickness = (finger_length / 4 ) / 1.8
            finger_radius = (finger_length / 5) / 1.8

            cv2.putText(frame, f'Length: {finger_length:.2f} mm', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Thickness: {finger_thickness:.2f} mm', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f'Radius: {finger_radius:.2f} mm', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def detect_angle(frame, hands, mp_hands, mp_drawing):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                tip_x = int(index_tip.x * frame.shape[1])
                tip_y = int(index_tip.y * frame.shape[0])
                mcp_x = int(index_mcp.x * frame.shape[1])
                mcp_y = int(index_mcp.y * frame.shape[0])
                dip_x = int(index_dip.x * frame.shape[1])
                dip_y = int(index_dip.y * frame.shape[0])

                dx = tip_x - dip_x
                dy = tip_y - dip_y
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad)

                cv2.putText(frame, f'Angle: {angle_deg:.2f} degrees', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                
                cv2.line(frame, (dip_x, dip_y), (tip_x, tip_y), (255,255,0), 2)
                cv2.line(frame, (mcp_x, mcp_y), (tip_x, tip_y), (255,0,0), 2)

        return frame
    

def generate_frames(mode):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            
            if mode == "dim":
                frame = detect_finger(frame, hands, mp_hands, mp_drawing)
            
            elif mode == "angle":
                frame = detect_angle(frame, hands, mp_hands, mp_drawing)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()