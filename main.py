import cv2
import mediapipe as mp
import numpy as np
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# --- CONFIGURAÇÕES MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Inicializa o modelo de mãos
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Estado da aplicação (para manter os desenhos na tela)
if "shapes" not in st.session_state:
    st.session_state.shapes = []

# --- LÓGICA DE PROCESSAMENTO ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    # Processamento Mediapipe
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Coordenadas dos dedos
            lm = hand_landmarks.landmark
            index_x, index_y = int(lm[8].x * w), int(lm[8].y * h)
            thumb_x, thumb_y = int(lm[4].x * w), int(lm[4].y * h)
            
            # Cálculo de distância para criar círculo (Pinça: Polegar + Indicador)
            distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
            
            if distance < 40:
                # Cria um novo círculo se a pinça for feita
                st.session_state.shapes.append({"x": index_x, "y": index_y, "r": 20})
                # Limita para não criar círculos infinitos (opcional)
                if len(st.session_state.shapes) > 50:
                    st.session_state.shapes.pop(0)

            # Desenha o esqueleto da mão
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Desenha as formas salvas
    for shape in st.session_state.shapes:
        cv2.circle(img, (shape["x"], shape["y"]), shape["r"], (0, 0, 255), -1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Air Drawing", layout="centered")
st.title("🖐️ Air Drawing Web")
st.write("Aproxime o **Indicador e o Polegar** para criar círculos na tela!")

if st.button("Limpar Desenho"):
    st.session_state.shapes = []

# Configuração de conexão para Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="air-drawing",
    mode=None,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)