# IA et apprentissage
 Travail projet IA L3

# Projet de Reconnaissance des Gestes de la Langue des Signes

## Introduction

Ce projet repose sur l’alphabet de la langue des signes (ASL) auquel j’ai ajouté deux signes personnalisés : un pour l’espace et un autre pour “fin” (END).

Le programme est écrit en Python, utilise OpenCV, MediaPipe et d’autres bibliothèques comme scikit-learn, et se base sur https://github.com/computervisioneng/sign-language-detector-python.

L’objectif de ce projet était de me familiariser avec les principes fondamentaux du machine learning, en passant par les étapes suivantes :

1. Création des données
2. Préparation des données
3. Entraînement du modèle

### Fonctionnalités Principales

- **Capture et analyse des gestes via la webcam**.
- **Détection et reconnaissance des formes et mouvements des mains**.
- **Traduction des gestes en texte et/ou vocal**.
- **Interface utilisateur interactive pour la visualisation des signes**.

## Etapes du Projet

### 1. Création des Données
La création des données se fait par la capture d’images de chaque signe via la webcam, en enregistrant chaque geste dans un répertoire correspondant au signe.

### 2. Prétraitement des Données
Les images capturées sont traitées pour extraire uniquement les repères (landmarks) de la main en utilisant la bibliothèque **MediaPipe**. Ces coordonnées sont ensuite normalisées et sauvegardées dans un fichier `data.pickle` pour l'entraînement du modèle.

### 3. Entraînement du Modèle
Le modèle de classification des gestes est entraîné à l’aide de **Random Forest**, un algorithme de machine learning. Le modèle apprend à associer les coordonnées des repères des mains aux signes de l’alphabet ASL et aux signes personnalisés.

### 4. Test du Modèle
Le modèle est testé en temps réel via un flux vidéo de la webcam. La détection des gestes est effectuée, et le signe reconnu est affiché à l'écran, avec une synthèse vocale lorsque le mot complet est détecté.

## Détails Techniques

- **MediaPipe** est utilisé pour la détection en temps réel des repères de la main.
- **Random Forest** est choisi pour la classification des gestes en raison de sa robustesse et de sa simplicité d'implémentation.
- **pyttsx3** est utilisé pour la synthèse vocale.

## Fonctionnalités de l'Interface Utilisateur (UI)

- Capture vidéo en direct analysée en temps réel.
- Affichage des repères de la main avec les coordonnées des articulations.
- Affichage du signe détecté et du mot en cours de formation.
- Commandes clavier :
  - `'q'` ou `'Esc'` : Quitter l'application.
  - `'r'` : Réinitialiser le mot en cours.

## Structure du Projet

Le projet contient les fichiers suivants :

- **Creation_images.py** : Script pour la capture d'images via la webcam.
- **Creation_dataset2.py** : Script pour organiser les images capturées en un dataset.
- **Entrainement_classification2.py** : Script pour entraîner le modèle Random Forest.
- **Test_classifyer.py** : Script pour tester le modèle en temps réel avec la webcam.

### Dossier `data`

Ce dossier contient 27 répertoires, chacun représentant un signe et contenant 100 images capturées par la webcam pour chaque signe.

### Fichier `requirements.txt`

Liste des dépendances Python requises pour exécuter le projet.

### Fichier `data.pickle`

Contient les données traitées et sérialisées pour l'entraînement du modèle.

### Fichier `model.p`

Modèle de classification entraîné, prêt à être utilisé pour la reconnaissance des signes.


## Améliorations Futures

1. **Optimisation du modèle** : Ajuster les hyperparamètres et la sélection des caractéristiques pour améliorer la précision.
2. **Évaluation sur un jeu de données extérieur** : Tester le modèle sur des images prises dans des conditions variées (éclairage, angle, etc.).
3. **Amélioration de l'interface utilisateur** : Ajouter des fonctionnalités interactives pour améliorer l'expérience utilisateur.

## Résultats

- **Précision de classification proche de 100%** sur le jeu de données utilisé.
- **Reconnaissance en temps réel** des 26 lettres de l'alphabet ASL et de 2 signes personnalisés.
- **Interface utilisateur fonctionnelle** pour visualiser les gestes et le mot en cours.

## Licence

Ce projet est sous licence MIT.

