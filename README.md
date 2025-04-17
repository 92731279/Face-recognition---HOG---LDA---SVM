# 🎯 Reconnaissance Faciale avec HOG, LDA et SVM

Ce projet implémente un système de **reconnaissance faciale** basé sur la **base de données FEI**, utilisant les méthodes de traitement d'image et de Machine Learning suivantes :

- Prétraitement d’images
- Détection de visages
- Extraction de caractéristiques avec **HOG**
- Réduction de dimension avec **LDA**
- Classification avec **SVM** et **GridSearchCV**

> 🔬 Réalisé par : Wided Mosbahi  
> 🎓 Master M1 ISIDS — Projet Python

---

## 📁 Données

Le projet utilise la base **FEI Face Database**, contenant **2800 images**. Les noms de fichiers sont structurés de manière à permettre une labellisation automatique (par personne).

---

## 🔧 Pipeline du projet

### 1. **Chargement des données**
- Lecture des images depuis un dossier.
- Attribution automatique des labels à partir des noms de fichiers.

### 2. **Prétraitement**
- Conversion en **niveaux de gris**
- **Redimensionnement** (100x100)
- **Binarisation** pour simplifier le traitement.

### 3. **Détection de visages**
- Utilisation du classificateur **Haarcascade** d'OpenCV pour extraire les visages.

### 4. **Extraction de caractéristiques (HOG)**
- Utilisation de `skimage.feature.hog` pour transformer chaque visage en vecteur de caractéristiques.

### 5. **Réduction de dimension (LDA)**
- Réduction de la dimensionnalité tout en conservant la séparabilité entre classes.
- Passage de plusieurs milliers de dimensions à 199.

### 6. **Classification (SVM + GridSearch)**
- Entraînement d’un modèle **SVM**.
- Optimisation des hyperparamètres `C` et `kernel` via **GridSearchCV**.
- Meilleur score atteint : **95.54%** avec `C=0.1` et `kernel='linear'`.

### 7. **Évaluation**
- **Classification report** : précision, rappel, F1-score
- **Matrice de confusion** pour visualiser les prédictions

---

## 📊 Résultats

- **Accuracy finale** : ~95.5%
- Forte concentration sur la diagonale dans la matrice de confusion (très peu d’erreurs)
- Bonne séparation entre les classes après LDA

---

## 🧠 Perspectives

- Intégration de **réseaux de neurones (CNN)** pour plus de robustesse
- Tests sur d'autres bases (ex : LFW, ORL, etc.)
- Meilleure tolérance aux variations de lumière, pose ou expression

---

## 🚀 Lancer le projet

### Prérequis
Installe les bibliothèques nécessaires :

```bash
pip install numpy pandas opencv-python scikit-learn matplotlib seaborn scikit-image
