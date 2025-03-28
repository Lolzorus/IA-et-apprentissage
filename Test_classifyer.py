import pickle  # Charger le modèle entraîné
import cv2  # OpenCV pour capturer la vidéo
import mediapipe as mp  # MediaPipe pour la détection des mains
import numpy as np  # NumPy pour manipuler les données
import time  # Gestion du temps pour la détection prolongée
import pyttsx3  # Module pour la synthèse vocale

# Charger le modèle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)  # 0 pour webcam intégrée, 1 pour externe

# Initialisation de MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configuration du modèle MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dictionnaire des labels avec ajout de l'espace et de la fin de mot
labels_dict = {i: chr(65 + i) for i in range(26)}  # 65 = 'A' en ASCII
labels_dict[26] = " "  # Espace
labels_dict[27] = "[END]"  # Fin du mot

# Initialisation du moteur de synthèse vocale
engine = pyttsx3.init()

# Variables pour construire un mot
current_letter = None
start_time = None
word = ""
previous_word = ""

# Boucle principale
while cap.isOpened():
    ret, frame = cap.read()  # Capture image webcam

    if not ret:
        print("Erreur : Impossible de capturer une image.")
        break

    H, W, _ = frame.shape  # Dimensions de l'image

    # Conversion en RGB pour MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)  # Détection des mains

    if results.multi_hand_landmarks:  # Si des mains sont détectées
        hand_landmarks = results.multi_hand_landmarks[0]  # Prendre la première main détectée

        # Dessiner la main détectée
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extraction et normalisation des coordonnées
        x_ = [lm.x for lm in hand_landmarks.landmark]
        y_ = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_), min(y_)
        data_aux = np.array([(lm.x - min_x, lm.y - min_y) for lm in hand_landmarks.landmark]).flatten()

        # Définition du rectangle englobant
        x1, y1 = int(min_x * W) - 10, int(min_y * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

        # Prédiction
        prediction = model.predict([data_aux])
        predicted_character = labels_dict[int(prediction[0])]

        # Gestion du temps de maintien
        if predicted_character == current_letter:
            if time.time() - start_time > 2:  # Maintenu > 2s
                if predicted_character == "[END]":
                    print(f"Mot finalisé : {word}")
                    engine.say(word)  # Lire le mot à haute voix
                    engine.runAndWait()
                    previous_word, word = word, ""  # Sauvegarde et reset du mot
                else:
                    word += predicted_character
                    print(f"Ajout au mot: {word}")
                start_time = time.time()
        else:
            current_letter, start_time = predicted_character, time.time()

        # Affichage de la prédiction
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    # Affichage du mot en cours et du mot précédent
    cv2.rectangle(frame, (10, 10), (W - 10, 60), (255, 255, 255), -1)
    cv2.putText(frame, word, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    cv2.rectangle(frame, (10, H - 60), (W - 10, H - 10), (200, 200, 200), -1)
    cv2.putText(frame, f"Mot precedent: {previous_word}", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Affichage de la fenêtre
    cv2.imshow('frame', frame)

    # Gestion des touches clavier
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break  # Quitter
    elif key == ord('r'):
        word = ""
        print("Mot reinitialise.")

# Libération des ressources
cap.release()
cv2.destroyAllWindows()