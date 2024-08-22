import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import time

# Charger le modèle YOLO que vous avez fine-tuné
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt')

# Configurer Streamlit
st.title("Détection de fatigue en temps réel")

# Démarrer la capture vidéo
cap = cv2.VideoCapture(0)

# Créer un conteneur vide pour l'image
frame_container = st.empty()

# Drapeau pour contrôler l'arrêt de la boucle
stop_flag = False

# Fonction pour démarrer la détection
def start_detection():
    global stop_flag
    stop_flag = False
    
    while cap.isOpened() and not stop_flag:
        ret, frame = cap.read()
        if not ret:
            st.write("Erreur: Impossible de lire la caméra")
            break
        
        # Détection de fatigue
        results = model(frame)
        
        # Rendre les résultats sur le cadre
        frame = np.squeeze(results.render())

        # Convertir l'image en format RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convertir en image PIL
        img = Image.fromarray(frame_rgb)
        
        # Afficher l'image dans le conteneur
        frame_container.image(img, use_column_width=True)

        # Ajouter un délai pour limiter le nombre de frames par seconde (facultatif)
        time.sleep(0.03)

# Fonction pour arrêter la détection
def stop_detection():
    global stop_flag
    stop_flag = True
    cap.release()
    cv2.destroyAllWindows()

# Boutons pour contrôler la détection dans le même cadre
col1, col2 = st.columns(2)

with col1:
    if st.button("Démarrer la détection"):
        start_detection()

with col2:
    if st.button("Arrêter la détection"):
        stop_detection()

