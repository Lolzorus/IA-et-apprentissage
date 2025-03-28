import os  # Module pour la gestion des fichiers et répertoires
import cv2  # OpenCV pour la capture et le traitement d'images

# Définition du répertoire où seront stockées les images capturées
DATA_DIR = './data'

# Vérifie si le répertoire de données existe, sinon le crée
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Définition du nombre de classes et du nombre d'images par classe
number_of_classes = 28  # Correspond aux 26 lettres de l'alphabet + 2 classes supplémentaires
dataset_size = 100  # Nombre d'images capturées par lettre (100 pour le projet mais plus le nombre est haut mieux c'est en theorie)

# Initialisation de la capture vidéo (webcam)
cap = cv2.VideoCapture(0)

# Vérifie si la caméra a été correctement détectée
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra")
    exit()

# Boucle pour capturer les images de chaque classe (lettre)
for j in range(number_of_classes):
    # Création d'un sous-dossier pour chaque classe s'il n'existe pas encore
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Vérification de la présence de fichiers existants
    if len(os.listdir(class_dir)) > 0:
        print(f"Des fichiers existent déjà pour la classe {j}, passage à la suivante...")
        continue

    print(f'Enregistrement des données pour la classe {j}')

    # Attente de l'utilisateur pour démarrer la capture d'images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo")
            exit()

        cv2.putText(frame, 'A pour lancer la capture', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('a'):
            break
        elif (cv2.waitKey(1) or 0xFF) == 27:  # Presser "ESC" pour quitter
            cap.release()
            cv2.destroyAllWindows()
            exit()

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo")
            exit()

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

# Libération des ressources après la capture des images
cap.release()
cv2.destroyAllWindows()
