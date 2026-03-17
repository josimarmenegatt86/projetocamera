import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Inicializa Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Lista de formas desenhadas
shapes = []  # cada forma será um dicionário {x, y, r}

# Variáveis de controle
drawing_mode = False
fingers_touching = False
selected_shape = None

# Funções auxiliares
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def fingers_are_open(landmarks):
    index_tip = landmarks[8]
    index_knuckle = landmarks[6]
    thumb_tip = landmarks[4]
    thumb_knuckle = landmarks[2]
    return index_tip.y < index_knuckle.y and thumb_tip.y < thumb_knuckle.y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Coordenadas dos dedos
            index_x, index_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            middle_x, middle_y = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)

            # Alternar modo de desenho com indicador + polegar
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
            if distance < 40:
                if not fingers_touching and fingers_are_open(hand_landmarks.landmark):
                    drawing_mode = not drawing_mode
                    print(f"Modo desenho {'ON' if drawing_mode else 'OFF'}")
                    time.sleep(0.2)
                fingers_touching = True
            else:
                fingers_touching = False

            # Se modo desenho ativo → cria círculo
            if drawing_mode:
                shapes.append({"x": index_x, "y": index_y, "r": 20})
                drawing_mode = False  # desliga para não criar infinitos círculos

            # Se dedo médio próximo de uma forma → mover
            if selected_shape is None:
                for shape in shapes:
                    if calculate_distance(middle_x, middle_y, shape["x"], shape["y"]) < shape["r"]:
                        selected_shape = shape
                        break
            else:
                # Atualiza posição da forma selecionada
                selected_shape["x"], selected_shape["y"] = middle_x, middle_y
                # Solta se dedo médio afastar
                if calculate_distance(middle_x, middle_y, selected_shape["x"], selected_shape["y"]) > 50:
                    selected_shape = None

            # Desenha landmarks da mão
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Desenha todas as formas
    for shape in shapes:
        cv2.circle(frame, (shape["x"], shape["y"]), shape["r"], (0, 0, 255), -1)

    cv2.imshow("Finger Drawing com Movimento", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
