import cv2
import mediapipe as mp
import numpy as np
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Configurações do Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.shapes = []  # Lista de círculos: {"x", "y", "r"}
        self.drawing_mode = False
        self.fingers_touching = False
        self.selected_shape = None

    def calculate_distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def fingers_are_open(self, landmarks):
        # Lógica simplificada para o navegador
        return landmarks[8].y < landmarks[6].y and landmarks[4].y < landmarks[2].y

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Coordenadas dos dedos
                index_x, index_y = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                thumb_x, thumb_y = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
                middle_x, middle_y = int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)

                # Alternar modo de desenho (Polegar + Indicador)
                distance = self.calculate_distance(index_x, index_y, thumb_x, thumb_y)
                if distance < 40:
                    if not self.fingers_touching:
                        self.drawing_mode = not self.drawing_mode
                        self.fingers_touching = True
                else:
                    self.fingers_touching = False

                # Lógica de Desenho
                if self.drawing_mode:
                    self.shapes.append({"x": index_x, "y": index_y, "r": 20})
                    self.drawing_mode = False

                # Lógica de Movimento (Dedo Médio)
                if self.selected_shape is None:
                    for shape in self.shapes:
                        if self.calculate_distance(middle_x, middle_y, shape["x"], shape["y"]) < shape["r"]:
                            self.selected_shape = shape
                            break
                else:
                    self.selected_shape["x"], self.selected_shape["y"] = middle_x, middle_y
                    if self.calculate_distance(middle_x, middle_y, self.selected_shape["x"], self.selected_shape["y"]) > 60:
                        self.selected_shape = None

                # Desenha esqueleto da mão
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Desenha as formas salvas
        for shape in self.shapes:
            cv2.circle(img, (shape["x"], shape["y"]), shape["r"], (0, 0, 255), -1)

        # Feedback visual do modo
        color = (0, 255, 0) if self.fingers_touching else (255, 255, 255)
        cv2.putText(img, f"Modo Desenho pronto", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Interface do Streamlit
st.title("🖐️ Air Drawing Web")
st.write("Aproxime o **Indicador e o Polegar** para criar um círculo. Use o **Dedo Médio** para arrastá-los.")

webrtc_streamer(
    key="hand-drawing",
    video_transformer_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Necessário para rodar na nuvem
    media_stream_constraints={"video": True, "audio": False},
)