import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données à partir du fichier pickle
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

# Récupération des données et des étiquettes du dictionnaire
data = np.asarray(data_dict['data'])  # Convertir les données en tableau numpy pour le traitement
labels = np.asarray(data_dict['labels'])  # Convertir les labels en tableau numpy

# Vérification de la présence de valeurs NaN dans les données
if np.isnan(data).sum() > 0:
    print("Attention : Des valeurs NaN ont été détectées dans les données !")
else:
    print("Aucune valeur NaN détectée dans les données.")

# Division des données en ensembles d'entraînement et de test (80% pour l'entraînement et 20% pour le test)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)
# Le paramètre `stratify=labels` permet de conserver la même distribution des classes dans les ensembles d'entraînement et de test.

# Initialisation du modèle Random Forest avec des hyperparamètres optimisés
model = RandomForestClassifier(n_estimators=200, max_depth=15, max_features='sqrt', random_state=42)
# `n_estimators=200` : Nombre d'arbres dans la forêt.
# `max_depth=15` : Profondeur maximale de chaque arbre, évite le sur-apprentissage.
# `max_features='sqrt'` : Utilisation d'un sous-ensemble des features pour chaque arbre pour améliorer la diversité.
# `random_state=42` : Permet de reproduire les résultats.

# Entraînement du modèle sur l'ensemble d'entraînement
model.fit(x_train, y_train)

# Prédiction des labels sur l'ensemble de test
y_predict = model.predict(x_test)

# Calcul de la précision du modèle (taux de bonnes prédictions)
accuracy = accuracy_score(y_test, y_predict)
print(f"Précision du modèle : {accuracy * 100:.2f}%")  # Affiche la précision en pourcentage

# Affichage du rapport de classification qui fournit des métriques détaillées : précision, rappel, f1-score
print("\nRapport de classification :\n", classification_report(y_test, y_predict))

# Calcul et affichage de la matrice de confusion pour visualiser les erreurs de classification
conf_matrix = confusion_matrix(y_test, y_predict)
plt.figure(figsize=(8, 6))  # Définir la taille de la figure pour la matrice de confusion
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=set(labels), yticklabels=set(labels))
# La matrice de confusion montre le nombre de bonnes et mauvaises classifications pour chaque classe.
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.title("Matrice de Confusion")
plt.show()

# Sauvegarde du modèle dans un fichier pickle pour un usage ultérieur
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Modèle sauvegardé avec succès !")
