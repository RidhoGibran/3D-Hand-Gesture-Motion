import cv2
import math
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Koordinat Titik Tengah Layar (Center)
    center_x = w // 2
    center_y = h // 2

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            
            # Deteksi posisi jari
            is_index_up = calculate_distance(hand_landmarks.landmark[8], wrist) > calculate_distance(hand_landmarks.landmark[5], wrist)
            is_middle_up = calculate_distance(hand_landmarks.landmark[12], wrist) > calculate_distance(hand_landmarks.landmark[9], wrist)
            is_ring_up = calculate_distance(hand_landmarks.landmark[16], wrist) > calculate_distance(hand_landmarks.landmark[13], wrist)
            is_pinky_up = calculate_distance(hand_landmarks.landmark[20], wrist) > calculate_distance(hand_landmarks.landmark[17], wrist)

            # 1. Telunjuk Saja -> Segitiga Hijau di Tengah Layar
            if is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
                pts = np.array([
                    [center_x, center_y - 40], 
                    [center_x - 40, center_y + 30], 
                    [center_x + 40, center_y + 30]
                ], np.int32)
                cv2.drawContours(frame, [pts], 0, (0, 255, 0), -1)
                cv2.putText(frame, "TELUNJUK (Segitiga Hijau)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 2. Gestur Peace (2 Jari) -> Teks "ME & YOU" di Tengah Layar
            elif is_index_up and is_middle_up and not is_ring_up and not is_pinky_up:
                # Menghitung ukuran teks agar posisi tepat di tengah
                text = "ME & YOU"
                font_scale = 1.5
                thickness = 3
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = center_x - (text_size[0] // 2)
                text_y = center_y + (text_size[1] // 2)

                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), thickness)
                cv2.putText(frame, "PEACE 2 JARI (ME & YOU)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            # 3. Kelingking Saja -> Lingkaran Kuning di Tengah Layar
            elif is_pinky_up and not is_index_up and not is_middle_up and not is_ring_up:
                cv2.circle(frame, (center_x, center_y), 40, (0, 255, 255), -1)
                cv2.putText(frame, "KELINGKING (Lingkaran Kuning)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # 4. Mengepal / Fist -> Lingkaran Merah di Tengah Layar
            elif not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
                cv2.circle(frame, (center_x, center_y), 50, (0, 0, 255), -1)
                cv2.putText(frame, "FIST (Lingkaran Merah)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # 5. Tangan Terbuka Penuh -> Persegi Oranye di Tengah Layar
            elif is_index_up and is_middle_up and is_ring_up and is_pinky_up:
                cv2.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), (255, 165, 0), -1)
                cv2.putText(frame, "OPEN (Persegi Oranye)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Motion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()