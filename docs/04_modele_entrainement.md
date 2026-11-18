# Étape 4 — Le modèle et l'entraînement

## 1. Le modèle en une image

```
 Œil gauche ? → miroir         (même orientation pour tous les yeux)
        │
 4 cartes 256×256 ──► ┌──────────────────────┐
  Axiale              │                      │──► 512 nombres ─┐
  Élévation ant.      │  ResNet18 PARTAGÉ    │──► 512 nombres ─┤
  Élévation post.     │  (pré-entraîné       │──► 512 nombres ─┼─► 2048 ─► tête (128) ─► 1 score
  Pachymétrie         │   sur ImageNet)      │──► 512 nombres ─┘                         │
                      └──────────────────────┘                                   sigmoid → 0 à 100 %
```

| Choix | Pourquoi |
|---|---|
| **4 cartes ensemble** | Le kératocône se voit sur plusieurs cartes à la fois (bombement + amincissement). Un médecin les regarde toutes. |
| **Extracteur partagé** (mêmes poids pour les 4 cartes) | 4 fois moins de paramètres → moins de surapprentissage avec ~900 yeux. |
| **ResNet18 pré-entraîné ImageNet** (*transfert d'apprentissage*) | Il sait déjà reconnaître formes, contours, dégradés. On l'adapte au lieu de tout apprendre depuis zéro. |
| **Miroir de l'œil gauche** | Les deux yeux sont symétriques : on aligne nasal/temporal pour que le modèle compare ce qui est comparable. |
| **Augmentation légère, sans changer les couleurs** | Petites rotations/translations = robustesse. Les couleurs codent les mesures : on n'y touche pas. |
| **`pos_weight`** | Il y a moins de malades que de sains : on donne plus de poids aux erreurs sur les malades. |
| **Suspect = positif** (`positive_labels`) | En dépistage, un cas suspect doit être signalé. Configurable. |

## 2. L'entraînement

| Mécanisme | Rôle |
|---|---|
| **AdamW** + **cosine annealing** | optimiseur standard ; le pas d'apprentissage diminue doucement |
| **Précision mixte** (GPU) | calculs en demi-précision, ~2× plus rapide, même résultat |
| **Sélection par perte de validation** | l'AUC de validation sature à 1,0 très vite sur CornOrb : la perte est plus informative pour choisir le meilleur modèle |
| **Arrêt anticipé** (5 époques sans progrès) | évite le surapprentissage et le temps perdu |
| **Graine fixe** | même config + même graine → même résultat (testé automatiquement) |
| **Jeu de test jamais lu** | il est réservé à l'évaluation finale (étape 5) : aucune décision n'est prise en le regardant |

## 3. Tout est piloté par configuration

```powershell
qarniya train configs/experiments/cornorb_resnet18.yaml
```
Comparer une autre architecture = un autre YAML (`cornorb_efficientnet.yaml`, 3 lignes différentes).
Ajouter une architecture = une fonction `@register_model("nom")` (même principe que les adaptateurs de données).

## 4. Chaque entraînement laisse une trace complète

```
outputs/runs/20261118_203000_cornorb_resnet18/
├── experiment.yaml   configuration exacte utilisée
├── dataset.yaml      configuration des données
├── model.pt          meilleur modèle
├── history.csv       perte, AUC, sensibilité, spécificité à chaque époque
└── summary.json      meilleure époque, métriques, version du logiciel, commit Git
```
**Le fichier `model.pt` contient les poids ET la préparation des images** (taille, miroir).
→ Impossible d'utiliser le modèle avec une autre préparation qu'à l'entraînement (bug classique en IA).
→ Rechargement sécurisé (`weights_only=True`) : un fichier modèle ne peut pas exécuter de code caché.

## 5. Les fichiers

| Fichier | Rôle |
|---|---|
| `models/preprocessing.py` | préparation des images, **partagée** entraînement / analyse |
| `models/multimap.py` | le réseau multi-cartes (backbones : resnet18, resnet34, efficientnet_b0) |
| `models/registry.py` | registre des modèles + `ModelSpec` |
| `models/checkpoint.py` | sauvegarde / rechargement sûrs et complets |
| `evaluation/metrics.py` | AUC, sensibilité, spécificité, matrice de confusion |
| `training/config.py` | `ExperimentConfig` (données + modèle + paramètres) |
| `training/dataset.py` | pont domaine → PyTorch |
| `training/trainer.py` | boucle générique (ne connaît ni le dataset ni l'architecture) |
| `training/experiment.py` | orchestre une expérience et écrit le dossier de résultats |

## 6. Tests ajoutés (79 au total)
- métriques vérifiées **à la main** sur un exemple, effet du seuil, AUC indéfinie ;
- préparation : forme, **miroir de l'œil gauche vérifié pixel à pixel**, augmentation ;
- modèles : sortie par backbone, mauvais nombre de cartes, noms inconnus ;
- sauvegarde : **prédictions identiques après rechargement**, préparation incluse ;
- de bout en bout : YAML → entraînement → dossier complet → modèle rechargeable ; **reproductibilité**.

## 7. Résultat de l'entraînement de référence (18/11/2026, Quadro RTX 4000)

`qarniya train configs/experiments/cornorb_resnet18.yaml` — ~2 min par époque.

| Époque | Perte entr. | Perte val. | AUC val. | Sensibilité | Spécificité | |
|---|---|---|---|---|---|---|
| 1 | 0,215 | 0,042 | 1,000 | 100 % | 99,3 % | ★ |
| 2 | 0,081 | 0,176 | 0,999 | 100 % | 92,2 % | |
| 3 | 0,098 | 0,108 | 0,995 | 98,8 % | 95,0 % | |
| **4** | 0,058 | **0,007** | **1,000** | **100 %** | **100 %** | ★ **meilleur** |
| 5–9 | → 0,0006 | 0,03–0,09 | ~1,000 | 97,5–98,8 % | 100 % | |

Arrêt anticipé après l'époque 9 (5 époques sans amélioration).

**Lecture :**
- après l'époque 4, la perte d'**entraînement** tombe presque à 0 alors que celle de **validation** ne baisse plus :
  le modèle commence à **apprendre par cœur** → l'arrêt anticipé a correctement gardé l'époque 4 ;
- 100 % sur la validation (222 yeux) ne veut PAS dire 100 % en réalité : CornOrb oppose surtout des yeux normaux à des
  kératocônes avérés, cas faciles. La vraie mesure sera le jeu de **test** (étape 5), jamais vu.
