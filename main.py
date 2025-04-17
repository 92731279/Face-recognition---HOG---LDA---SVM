import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Étape 1 : Chargement des images dans un DataFrame ===
image_dir = 'C:\\Users\\wided\\OneDrive\\Desktop\\1misids\\s2\\python\\FEI BD'
file_list = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]

df = pd.DataFrame({'filename': file_list})
df['image_path'] = df['filename'].apply(lambda f: os.path.join(image_dir, f))

# Regroupement des images par personne (label = personne)
df['label'] = df['filename'].apply(lambda f: f.split('-')[0])  # Regrouper par personne, par exemple '1'

df['image'] = df['image_path'].apply(cv2.imread)

print(f"✅ Étape 1 : {len(df)} images chargées.")
print("🔍 Exemple de labels :", df['label'].unique()[:5])
plt.imshow(cv2.cvtColor(df['image'][0], cv2.COLOR_BGR2RGB))
plt.title("Image originale")
plt.axis('off')
plt.show()

# === Étape 2 : Prétraitement ===
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100))
    _, binary = cv2.threshold(resized, 127, 1, cv2.THRESH_BINARY)
    return binary.astype(np.float32)

df['preprocessed'] = df['image'].apply(preprocess_image)

print("✅ Étape 2 : Prétraitement terminé (niveaux de gris + binarisation).")
plt.imshow(df['preprocessed'][0], cmap='gray')
plt.title("Image prétraitée")
plt.axis('off')
plt.show()

# === Étape 3 : Détection de visages ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image):
    image = (image * 255).astype(np.uint8)
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    for (x, y, w, h) in faces:
        return cv2.resize(image[y:y+h, x:x+w], (64, 64))
    return cv2.resize(image, (64, 64))

df['face'] = df['preprocessed'].apply(detect_face)

print("✅ Étape 3 : Visages détectés.")
plt.imshow(df['face'][0], cmap='gray')
plt.title("Visage détecté")
plt.axis('off')
plt.show()

# === Étape 4 : Extraction de caractéristiques HOG ===
def extract_hog_features(image):
    return hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               orientations=9, block_norm='L2-Hys')

df['features'] = df['face'].apply(extract_hog_features)

X = np.stack(df['features'].to_numpy())
le = LabelEncoder()
y = le.fit_transform(df['label'])

print("✅ Étape 4 : Extraction des caractéristiques HOG terminée.")
print("🔢 Exemple de vecteur de caractéristiques HOG :", X[0][:10])  # 10 premières valeurs

# === Étape 5 : Réduction dimensionnelle avec LDA ===
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)

print(f"✅ Étape 5 : Réduction de dimensions avec LDA -> Dimensions : {X_lda.shape}")

# === Étape 6 : SVM avec GridSearch ===
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\n✅ Étape 6 : Entraînement du SVM terminé avec GridSearch.")
print("🔍 Meilleur score :", grid.best_score_)
print("⚙️ Meilleurs hyperparamètres :", grid.best_params_)

# === Étape 7 : Évaluation ===
y_pred = grid.predict(X_test)
print("\n✅ Étape 7 : Évaluation du modèle.")
print("📊 Rapport de classification :\n", classification_report(y_test, y_pred))
print("🎯 Accuracy :", accuracy_score(y_test, y_pred))

# === Matrice de confusion ===
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=False, fmt='d', cmap='Blues')
plt.title("Matrice de confusion")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.show()
