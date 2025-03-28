import os
import pickle
import mediapipe as mp
import cv2
import concurrent.futures

# Initialisation de MediaPipe Hands
# MediaPipe est une bibliothèque pour la détection de mains, utilisée ici pour analyser les images.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Définition du répertoire contenant les images
DATA_DIR = './data'  # Répertoire où se trouvent les images à traiter

# Initialisation des listes pour stocker les données et les étiquettes
data = []  # Liste pour stocker les coordonnées normalisées des landmarks de la main
labels = []  # Liste pour stocker les étiquettes correspondant à chaque image (représente la classe de la main)

# Vérification de l'existence du dossier de données
if not os.path.exists(DATA_DIR):  # Si le dossier n'existe pas
    print(f"[ERREUR] Le répertoire {DATA_DIR} n'existe pas. Vérifiez le chemin.")  # Affiche un message d'erreur
    exit()  # Arrête l'exécution du programme

def process_image(img_file, dir_):
    """Traite une image et extrait les coordonnées des landmarks."""
    # Vérifie si l'image existe et est un fichier valide
    if not os.path.isfile(img_file):
        print(f" [AVERTISSEMENT] {img_file} n'est pas un fichier valide, ignoré.")
        return None  # Si ce n'est pas un fichier valide, retourne None

    # Lecture de l'image avec OpenCV
    img = cv2.imread(img_file)
    if img is None:
        print(f" [AVERTISSEMENT] Impossible de lire {img_file}, image ignorée.")
        return None  # Si l'image ne peut pas être lue, retourne None

    try:
        # Conversion de l'image en RGB (MediaPipe attend un format RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processus de détection des landmarks des mains avec MediaPipe
        results = hands.process(img_rgb)

        # Si des mains ont été détectées
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraction des coordonnées des landmarks de la main (x, y)
                x_, y_ = zip(*[(lm.x, lm.y) for lm in hand_landmarks.landmark])
                if not x_ or not y_:
                    print(f" [AVERTISSEMENT] Aucune coordonnée détectée pour {img_file}, image ignorée.")
                    return None  # Si aucune coordonnée n'est trouvée, retourne None

                # Normalisation des coordonnées des landmarks par rapport à la main détectée
                min_x, min_y = min(x_), min(y_)  # Détermine les coordonnées minimales pour normalisation
                data_aux = [(x - min_x, y - min_y) for x, y in zip(x_, y_)]  # Normalisation des coordonnées

                # Retourne les coordonnées normalisées aplanies et l'étiquette de la classe
                return [coord for pair in data_aux for coord in pair], dir_

    except Exception as e:
        # Si une erreur survient pendant le traitement de l'image, l'exception est capturée
        print(f" [ERREUR] Problème lors du traitement de {img_file} : {e}")

    return None  # Si aucune main n'est détectée ou en cas d'erreur, retourne None

# Parcours des sous-dossiers représentant les classes (lettres)
image_tasks = []  # Liste pour stocker les tâches futures
with concurrent.futures.ThreadPoolExecutor() as executor:  # Utilisation d'un ThreadPoolExecutor pour traiter les images en parallèle
    # Parcours des sous-dossiers du répertoire de données
    for dir_ in sorted(os.listdir(DATA_DIR), key=lambda x: int(x) if x.isdigit() else x):
        dir_path = os.path.join(DATA_DIR, dir_)  # Chemin complet du dossier de classe

        # Vérifie si l'élément est bien un dossier
        if not os.path.isdir(dir_path):
            continue  # Si ce n'est pas un dossier, on passe au suivant

        print(f"[INFO] Traitement de la classe '{dir_}'...")  # Affiche le message de début de traitement pour la classe

        # Parcours des images dans le sous-dossier de la classe
        for img_path in os.listdir(dir_path):
            img_file = os.path.join(dir_path, img_path)  # Chemin complet de l'image
            # Soumet chaque tâche de traitement d'image au ThreadPoolExecutor
            image_tasks.append(executor.submit(process_image, img_file, dir_))

    # Attente que toutes les tâches soient terminées
    for future in concurrent.futures.as_completed(image_tasks):
        result = future.result()  # Récupère le résultat de chaque tâche
        if result:
            # Si un résultat est obtenu, on ajoute les données et les labels aux listes respectives
            data.append(result[0])
            labels.append(result[1])

# Sauvegarde des données dans un fichier pickle
try:
    with open('data.pickle', 'wb') as f:  # Ouvre un fichier en mode écriture binaire
        pickle.dump({'data': data, 'labels': labels}, f)  # Sauvegarde les données et labels dans le fichier pickle
    print("[SUCCÈS] Données sauvegardées avec succès dans 'data.pickle'.")  # Affiche un message de succès

except Exception as e:
    # Si une erreur survient pendant la sauvegarde des données, l'exception est capturée
    print(f"[ERREUR] Échec de la sauvegarde des données : {e}")
