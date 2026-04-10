# Forest Cover Change Detection

Pipeline de prototypage pour detecter et quantifier une evolution de la couverture vegetale entre deux dates a partir des images pretraitees du projet.

## Structure

- `src/preprocessing.py` : chargement des images et extraction des features
- `src/clustering.py` : CAH, K-means, identification automatique de la vegetation
- `src/analysis.py` : post-traitement, quantification et generation des sorties
- `data/` : images source brutes
- `outputs/preprocessed_images/` : images pretraitees fournies pour l'analyse
- `outputs/` : resultats produits par le pipeline
- `notebooks/` : demonstration / essais

## Features retenues

Le pipeline respecte les contraintes du sujet :

- RGB
- HSV
- ratio vert `G / (R + G + B)`
- moyenne locale
- variance locale

## Execution

Depuis la racine du projet :

```bash
python -m src.analysis
```

Pour ce cas d'etude :

- `outputs/preprocessed_images/16_t0_preprocessed.png` correspond a l'etat avant
- `outputs/preprocessed_images/17_t1_preprocessed.png` correspond a l'etat apres

Le pipeline peut toujours etre relance sur d'autres images avec :

```bash
python -m src.analysis --t0 chemin/vers/image_avant.png --t1 chemin/vers/image_apres.png --output-dir outputs
```

Les sorties seront generees dans `outputs/` :

- `dendrogram_t0.png`
- `dendrogram_t1.png`
- `segmentation_t0.png`
- `segmentation_t1.png`
- `vegetation_mask_t0.png`
- `vegetation_mask_t1.png`
- `deforestation_map.png`
- `analysis_summary.png`
- `metrics.json`

## Hypothese importante

Le pipeline suppose que `t0` et `t1` representent la meme zone et sont comparables spatialement. Si les deux images ne sont pas bien alignees ou ne couvrent pas la meme scene, la quantification perd son sens.
