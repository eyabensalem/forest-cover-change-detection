# 🌳 Forest Cover Change Detection

## 🎯 Objectif du projet

Ce projet vise à **détecter et analyser les changements de couverture du sol entre deux images satellites prises à des dates différentes (`t0` et `t1`)**, afin d’identifier une éventuelle **déforestation**.

L’objectif final est de :

* détecter les zones de changement
* localiser ces changements
* préparer la quantification de la perte de végétation

---

## 📂 Structure du projet

```text
forest-cover-change-detection/
│
├── data/
│   ├── t0.png
│   └── t1.png
│
├── outputs/
│
├── src/
│   ├── io_utils.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── features.py
│   ├── main.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Créer un environnement virtuel :

```bash
python -m venv venv
```

Activer l’environnement :

### Windows

```bash
venv\Scripts\activate
```

### Mac / Linux

```bash
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## ▶️ Exécution

Depuis la racine du projet :

```bash
python src/main.py
```

Les résultats seront générés dans le dossier `outputs/`.

---

## 🧠 Méthodologie

### 1. Analyse initiale

* Chargement des images `t0` et `t1`
* Visualisation
* Analyse des histogrammes RGB

👉 Permet de comprendre les différences globales entre les deux dates.

## 🖼️ Images initiales

![Images initiales](outputs/01_images_initiales.png)
---

### 2. Prétraitement

* Débruitage médian
* Amélioration du contraste local (CLAHE)

👉 Objectif : améliorer la qualité visuelle et rendre les structures plus discriminantes.

## ⚙️ Prétraitement

![Images prétraitées](outputs/04_images_pretraitees.png)
---

### 3. Représentation des pixels

* RGB
* HSV
* Ratio vert

👉 Le **ratio vert** permet de mettre en évidence la végétation.

---

### 4. Features locales

* Moyenne locale
* Variance locale

👉 Ces descripteurs capturent :

* la structure
* la texture
* les contours (notamment utiles pour détecter l’urbanisation)

---

### 5. Détection de changement

* Calcul de la différence entre `t0` et `t1`
* Combinaison de plusieurs features :

  * ratio vert
  * moyenne locale
  * variance locale

👉 Permet de construire une **carte de changement**
## 🌿 Ratio vert

![Ratio vert t0](outputs/05_ratio_vert_t0.png)
![Ratio vert t1](outputs/06_ratio_vert_t1.png)
## 🔥 Carte de changement

![Change map](outputs/11_change_map.png)
---

### 6. Génération du masque

* Seuillage automatique
* Nettoyage morphologique
* Suppression des petites zones

👉 Permet d’isoler les zones de changement significatif
## 🎯 Détection finale

![Change mask](outputs/13_change_mask_nettoye.png)
---

## 📊 Résultats

Le dossier `outputs/` contient :

* images initiales
* histogrammes RGB
* images prétraitées
* ratio vert
* moyenne locale
* variance locale
* carte de changement
* masque de changement final

---

## 📌 Interprétation

Les résultats montrent une **réduction des zones végétalisées entre `t0` et `t1`**, accompagnée de l’apparition de structures (routes, bâtiments).

👉 Cela indique une **urbanisation et une perte de végétation**, cohérente avec un phénomène de déforestation.

---

## ⚠️ Limites

* Absence d’alignement parfait entre les images
* Sensibilité aux conditions d’éclairage
* Détection basée sur des heuristiques simples

👉 Des améliorations possibles :

* recalage des images
* normalisation radiométrique
* segmentation avancée (KMeans)

---

## 🚀 Suite du projet

La suite consiste à :

* appliquer une **classification (KMeans)**
* identifier automatiquement la végétation
* calculer la surface végétalisée
* quantifier la perte de végétation

---

## 🧠 Conclusion

Ce projet met en place un pipeline complet de traitement d’images pour :

* extraire des caractéristiques pertinentes
* comparer deux dates
* détecter les changements de couverture du sol

👉 La végétation est utilisée comme indicateur principal pour analyser la déforestation.

