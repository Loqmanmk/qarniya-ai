# Qarniya AI — Guide pas à pas

Dépistage précoce du kératocône à partir des 4 cartes de topographie cornéenne.
Équipe NeuroLens · Hackathon Nexus HealthTech 2026 (EHEIM Oujda).

---

## 0. Vue d'ensemble

```
4 cartes PNG d'un œil ──► ResNet18 partagé (×4) ──► fusion ──► score de risque (0–100 %)
 (Axial, Anterior,                                          └─► Grad-CAM : carte de chaleur
  Posterior, Pachymetry)                                         des zones suspectes
```

| Fichier | Rôle |
|---|---|
| `src/config.py` | chemins et hyperparamètres |
| `src/dataset.py` | lecture du CSV et des 4 images, découpage train/val/test |
| `src/model.py` | le réseau de neurones |
| `src/train.py` | entraînement |
| `src/evaluate.py` | évaluation sur le jeu de test + courbe ROC |
| `src/baseline_tabular.py` | comparaison avec un modèle simple sur les chiffres (Kmax…) |
| `src/predict.py` | score + carte de chaleur pour un œil |
| `src/app.py` | interface web de démo (analyse + suivi de progression) |

---

## 1. Outils à installer

| Outil | Pourquoi | Où |
|---|---|---|
| **Python 3.12** | langage du projet (déjà installé chez vous) | python.org |
| **Pilote NVIDIA** | utiliser le GPU Quadro RTX 4000 (déjà installé) | — |
| **VS Code** + extension Python | éditer et lancer le code | code.visualstudio.com |
| **Git** (optionnel) | sauvegarder les versions du code | git-scm.com |

Bibliothèques Python (installées dans un environnement virtuel `.venv`) :
PyTorch + torchvision (deep learning, GPU), pandas, scikit-learn, Pillow, OpenCV, matplotlib, tqdm, Gradio (interface web).

### Commandes (PowerShell, dans le dossier `Quaraniya`)

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\Activate.ps1
```
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
```powershell
pip install -r requirements.txt
```

Vérifier que le GPU est vu :
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
→ doit afficher `True Quadro RTX 4000`.

> Si PowerShell refuse `Activate.ps1` : `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> (ou utilisez directement `.\.venv\Scripts\python.exe` à la place de `python`).

---

## 2. Les données (déjà téléchargées)

Dataset **CornOrb** (Zenodo, licence CC BY 4.0, Orbscan 3, Algérie) :
<https://zenodo.org/records/20542091>

```
data/cornorb/
├── clinical_data_and_labels.csv     1 454 lignes (une par œil)
└── ORBSCAN_Dataset/<patient>/<OD|OS>/<patient>_<œil>_<Axial|Anterior|Posterior|Pachymetry>.png
```

- 1 407 yeux ont leurs images. Le code ignore automatiquement 2 lignes sans images et
  46 yeux présents **deux fois** dans le CSV avec des valeurs différentes (impossible de savoir
  quelle ligne correspond aux images) → environ 1 360 yeux utilisés.
- `label` : 0 = normal, 1 = kératocône.
- `split_80_20` + `fold` : découpage officiel **par patient** → on l'utilise tel quel
  (train = folds 2–5, validation = fold 1, test = 291 yeux jamais vus).
- Les yeux gauches (OS) sont retournés en miroir pour avoir nasal/temporal au même endroit que l'œil droit.

**Citez le dataset** dans vos slides : Lazouni et al., *CornOrb*, Zenodo 2025, DOI 10.5281/zenodo.17121301.

---

## 3. Étapes de réalisation

### Étape 1 — Point de comparaison (1 minute)
```powershell
python src/baseline_tabular.py
```
Régression logistique sur Kmax, épaisseur, astigmatisme… Donne une AUC de référence.
Votre modèle sur images doit faire **au moins aussi bien**, et il apporte en plus la localisation (Grad-CAM).

### Étape 2 — Test rapide du pipeline (2–3 minutes)
```powershell
python src/train.py --epochs 2 --limit 120
```
Sert seulement à vérifier qu'il n'y a pas d'erreur. Les scores n'ont pas de sens ici.

### Étape 3 — Entraînement complet (~15–30 min sur le GPU)
```powershell
python src/train.py
```
À chaque époque vous voyez `val_AUC`, `sens` (sensibilité), `spec` (spécificité).
Le meilleur modèle est sauvegardé dans `outputs/qarniya_model.pt`.

### Étape 4 — Évaluation sur le test
```powershell
python src/evaluate.py
```
Produit `outputs/test_metrics.json`, `outputs/test_predictions.csv`, `outputs/roc_test.png` (pour les slides).

### Étape 5 — Carte de chaleur sur un œil
```powershell
python src/predict.py --folder data/cornorb/ORBSCAN_Dataset/9C6SM/OD
```
Écrit `outputs/9C6SM_OD_gradcam.png` : les 4 cartes avec les zones qui ont fait monter le score.

### Étape 6 — Démo web
```powershell
python src/app.py
```
Ouvrez <http://127.0.0.1:7860>.
- Onglet **Analyse d'un œil** : déposez les 4 cartes → score + niveau (Faible / À surveiller / Élevé) + chaleur.
- Onglet **Suivi de progression** : deux examens du même œil → évolution du score.

---

## 3 bis. Version Google Colab (sans rien installer)

Fichier : **`Qarniya_AI_Colab.ipynb`** — il contient tout (téléchargement des données + code + démo).

1. Allez sur <https://colab.research.google.com> et connectez-vous avec un compte Google.
2. **Fichier → Importer un notebook → Importer** → choisissez `Qarniya_AI_Colab.ipynb`.
3. **Exécution → Modifier le type d'exécution → GPU T4 → Enregistrer.**
4. Exécutez les cellules une par une, de haut en bas (Maj + Entrée), ou **Exécution → Tout exécuter**
   (sauf que la démo, étape 10, bloque : arrêtez-la avec ■ pour continuer).
5. Étape 10 : cliquez sur le lien `https://….gradio.live` → démo ouverte dans le navigateur,
   partageable avec le médecin ou le jury.
6. Étape 11 : télécharge `qarniya_resultats.zip` (modèle + figures). **Colab efface tout à la fermeture.**

Bon à savoir : Colab gratuit coupe après ~90 min d'inactivité ou quelques heures d'usage GPU par jour.
Si le GPU est refusé, réessayez plus tard ou utilisez votre PC (sections 1 à 3).
Ne mettez **pas** de vraies images patients dans Colab sans l'accord du médecin (loi 09-08) :
les données passent sur les serveurs de Google, et le lien gradio.live est public.

## 4. Comment tester (et quoi vérifier)

| Test | Comment | Résultat attendu |
|---|---|---|
| Installation | commande `torch.cuda.is_available()` | `True` |
| Pipeline | `train.py --epochs 2 --limit 120` | pas d'erreur, fichier `.pt` créé |
| Performance | `evaluate.py` | AUC test ≥ baseline tabulaire ; sensibilité élevée (> 0,9 visé) |
| Cohérence médicale | `predict.py` sur 5 KC et 5 normaux | la chaleur doit tomber sur le cône (souvent inféro-temporal), pas sur le texte ou les bords |
| Cas faciles/difficiles | trier `test_predictions.csv` par `prob_kc` | regarder les erreurs (FN = KC manqués, les plus graves) |
| Démo | `app.py` avec des images du dossier test | réponse en < 2 s, affichage correct |
| Données du médecin | mettre ses PNG dans un dossier par œil, lancer `predict.py` | vérifier avec lui si les zones signalées sont cohérentes |

**Règles d'or :**
- Ne jamais régler le modèle en regardant le **test** : utilisez la validation. Le test ne sert qu'une fois, à la fin.
- En médecine, un kératocône manqué (FN) est pire qu'une fausse alerte (FP) : présentez la **sensibilité**.
- Si la chaleur tombe sur les chiffres imprimés par l'appareil, le modèle « triche » : dites-le honnêtement et c'est une piste d'amélioration.

---

## 4 bis. Résultats obtenus (18/11/2026, Quadro RTX 4000, ~1 360 yeux)

Jeu de test : 265 yeux (174 normaux, 91 kératocônes), jamais vus à l'entraînement.

| Modèle | AUC | Sensibilité | Spécificité | KC manqués |
|---|---|---|---|---|
| Baseline tabulaire (Kmax, épaisseur…) | 0,984 | 81,3 % | 100 % | 17 |
| **Qarniya AI (4 cartes, ResNet18)** | **0,998** | **95,6 %** | **98,9 %** | **4** |

→ Le modèle sur images détecte **13 kératocônes de plus** que les chiffres seuls.
Grad-CAM : sur un KC, la chaleur se concentre sur le cône dans les cartes d'élévation antérieure/postérieure
(voir `outputs/PN34D_OS_gradcam.png`), pas sur le texte imprimé.

Attention : la validation atteint AUC = 1,000 dès l'époque 1 → CornOrb oppose surtout des yeux normaux
à des kératocônes avérés, cas faciles. Le vrai défi (formes frustes / suspects) n'est pas dans ce dataset.

## 5. Limites à annoncer au jury

1. CornOrb n'a que 2 classes (normal / kératocône), **pas de « suspect »** → le niveau « À surveiller » est une zone de score intermédiaire, non validée cliniquement.
2. Pas d'examens répétés dans CornOrb → l'onglet progression est une démonstration ; il faut les données longitudinales du médecin pour le valider.
3. Un seul appareil (Orbscan 3) : sur Pentacam ou Sirius, les couleurs diffèrent → il faudra ré-entraîner (fine-tuning) avec quelques images de cet appareil.
4. Population algérienne, proche de la population marocaine, mais à valider localement.

---

## 6. Améliorations si vous avez le temps

- **Fusion images + chiffres** : ajouter Kmax et épaisseur minimale à l'entrée de la tête du modèle.
- **Validation croisée 5 folds** (colonne `fold`) pour un résultat plus robuste.
- **Masquer le texte et la grille** des images (plus de risque que le modèle lise les chiffres).
- **Seuil ajusté** pour viser 95 % de sensibilité.
- **Fine-tuning** sur les images du médecin dès qu'elles arrivent.
