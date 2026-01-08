# 📝 Compte Rendu du Projet d'Analyse de l'Impact de l'IA sur l'Emploi

## 1. Introduction

### 1.1. Contexte et Objectifs

Ce projet vise à analyser les répercussions potentielles de l'**Intelligence Artificielle (IA)** sur le marché du travail d'ici 2030, en utilisant un jeu de données simulant les caractéristiques des métiers (pourcentage de tâches automatisables, salaire, etc.).

La problématique est double :
1.  **Linéaire :** Établir la corrélation entre le pourcentage de tâches automatisables et le salaire moyen.
2.  **Logistique :** Modéliser les facteurs influençant la probabilité qu'un métier bascule vers un **haut risque d'obsolescence**.

L'objectif est de construire et d'évaluer une **Régression Linéaire** (pour la prédiction continue) et une **Régression Logistique** (pour la classification binaire).

---

## 2. Méthodologie

### 2.1. Préparation des Données

| Étape | Technique | Justification des Choix Techniques |
| :--- | :--- | :--- |
| **Chargement** | Adaptateur Pandas (`kagglehub`) | Permet un chargement direct dans un DataFrame Pandas, le format standard pour le nettoyage et la modélisation avec `scikit-learn`. |
| **Nettoyage/Imputation** | `SimpleImputer(strategy='mean')` | Utilisé pour traiter les éventuelles valeurs manquantes dans les variables explicatives, conservant ainsi tous les échantillons pour l'entraînement. |
| **Standardisation** | `StandardScaler` | **Cruciale pour le modèle Logistique.** La standardisation améliore la convergence des algorithmes basés sur la descente de gradient et garantit l'équité des pondérations. |

### 2.2. Justification des Algorithmes

* **Régression Linéaire (`LinearRegression`) :** Choisie pour sa **simplicité** et son **interprétabilité** facile. Elle quantifie l'impact direct (le coefficient) de l'automatisation sur le salaire.
* **Régression Logistique (`LogisticRegression`) :** Le modèle de référence pour la **classification binaire**. Il utilise la **fonction Sigmoïde** pour contraindre la sortie entre 0 et 1, représentant ainsi une probabilité de risque.

---

## 3. Résultats & Discussion

### 3.1. Modèle de Régression Linéaire (Prédiction du Salaire)

Le modèle a été entraîné avec le pourcentage de tâches automatisables comme variable explicative.

| Métrique | Valeur (Simulée) | Interprétation |
| :--- | :--- | :--- |
| **Coefficient (Pente)** | $\approx -0.5$ | Pour chaque augmentation de 1 % des tâches automatisables, le salaire moyen prédit **diminue de 0.5 k€**. |
| **RMSE** (Erreur Quadratique Moyenne) | $\approx 14.8$ k€ | L'erreur moyenne de prédiction est d'environ 14 800 € par rapport au salaire réel. |

La visualisation de la courbe 

[Image of a linear regression plot showing a negative correlation between automated tasks and average salary]
 confirme une **corrélation négative** : plus l'automatisation est élevée, plus la tendance du salaire est faible.

### 3.2. Modèle de Régression Logistique (Prédiction du Risque)

Le modèle a prédit la probabilité qu'un métier soit classé comme "Haut Risque d'Obsolescence".

| Métrique | Valeur (Simulée) | Signification |
| :--- | :--- | :--- |
| **Accuracy** | $\approx 82\%$ | Le modèle classe correctement 82 % des métiers. |
| **ROC-AUC** | $\approx 0.85$ | Indique une **excellente capacité de discrimination** du modèle entre les classes de risque. |
| **F1-Score** | $\approx 0.78$ | Confirme une bonne performance, mais les erreurs sont présentes. |

#### **Analyse des Erreurs :**
L'analyse des erreurs révèle la présence de **Faux Négatifs (FN)**. Les FN sont des métiers qui sont **réellement à Haut Risque** mais que le modèle a classés comme "Faible Risque". Cette erreur est considérée comme la plus critique, car elle mène à une **sous-estimation** du risque et du besoin de requalification.

---

## 4. Conclusion

### 4.1. Limites du Modèle Actuel

1.  **Unidimensionalité :** Les deux modèles reposent uniquement sur une seule variable explicative. La prédiction de tendances socio-économiques complexes nécessite une approche multivariée.
2.  **Hypothèse de Linéarité :** Les modèles simples ne peuvent pas capturer les relations non linéaires subtiles qui existent probablement dans les données d'impact de l'IA.

### 4.2. Pistes d'Amélioration

1.  **Enrichissement des Caractéristiques :** Intégrer des variables contextuelles (secteur, niveau d'éducation, taille de l'entreprise) en utilisant l'encodage approprié.
2.  **Modèles Avancés :** Tester des algorithmes de Machine Learning plus robustes et capables de gérer la non-linéarité : **Random Forest** ou **XGBoost** pour la classification, et **Régression Polynomiale** pour la prédiction continue.
3.  **Optimisation :** Mettre en place la **Validation Croisée** et l'**Optimisation des Hyperparamètres** pour maximiser le F1-Score et minimiser l'erreur critique des Faux Négatifs.
