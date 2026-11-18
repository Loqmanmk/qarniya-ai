# Architecture du projet

> Rédigé à l'étape 2, **mis à jour à l'étape 11** (état final du hackathon).

## 1. Vue d'ensemble : une architecture en couches

```
 Déploiement   Docker (non-root, lecture seule) ─── qarniya serve ─── 127.0.0.1:8765
                                    │
            ┌───────────────────────┴──────────────────────────────────────┐
 Interfaces │ api/static/ (page web du médecin)   api/ (FastAPI, /docs)    │  ← comment on parle
            │ cli.py : data inspect · train · evaluate · prepare · analyse · serve │     au logiciel
            └───────────────────────┬──────────────────────────────────────┘
                                    ▼
 Services   ┌──────────────────────────────────────────────────────────────┐
            │ analysis.py   analyser un œil (qualité → score + chaleur → nouveauté) │  ← les cas d'usage
            │ progression.py comparer 2 examens · clinical.py avis des chiffres     │    (user stories)
            │ report.py     rapport PDF · triage_order : dépister un lot            │
            └───────┬──────────────────┬───────────────────┬───────────────┘
                    ▼                  ▼                   ▼
 Technique  ┌────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
            │ data/          │ │ models/          │ │ explainability/      │  ← les « détails »
            │  adapters/     │ │  multimap, registry│ │  gradcam, rendering  │    remplaçables
            │  splitting     │ │  predictor       │ │ evaluation/          │
            │  validation    │ │  novelty, checkpoint│ │  metrics, thresholds,│
            │                │ │ training/ (callbacks)│ │  bootstrap, baseline │
            └───────┬────────┘ └────────┬─────────┘ └──────────┬───────────┘
                    └───────────────────┼──────────────────────┘
                                        ▼
 Domaine    ┌──────────────────────────────────────────────────────────────┐
            │ CornealExam · CornealMaps · RiskAssessment · RiskThresholds · │  ← le cœur métier,
            │ ProgressionAssessment · OpinionComparison                      │    pur Python
            └──────────────────────────────────────────────────────────────┘
```

### Le trajet d'une analyse (de la page web au résultat)
```
page web ──POST /api/analyses──► api/app.py : vérifie et ré-encode les 4 images (dossier temporaire)
   ▲                                   │
   │                                   ▼
   │                  services/analysis.py
   │                    ① data/validation.py   qualité ── refus ──► « Analyse impossible »
   │                    ② models/predictor.py  + explainability/gradcam.py → score, chaleur
   │                    ③ models/novelty.py    image inhabituelle ? → « score non fiable »
   │                    ④ services/clinical.py avis des chiffres → domain OpinionComparison
   │                                   │
   └──────── JSON (score, niveau, 4 chaleurs, alertes) ◄─┘   dossier temporaire supprimé
```

**Règle des dépendances :** les flèches vont toujours **vers le domaine**.
Le domaine ne connaît ni PyTorch, ni pandas, ni le web. → Ce n'est pas une promesse : le test
`tests/unit/test_architecture.py` le **vérifie automatiquement** à chaque exécution.

## 2. Pourquoi cette architecture ? (ce qu'on dit au jury)

| Besoin | Réponse de l'architecture |
|---|---|
| Changer de **dataset** (CornOrb → données du médecin → Pentacam) | Un nouvel **adaptateur** dans `data/adapters/` + un fichier YAML. Le reste ne bouge pas. |
| Changer de **modèle** (ResNet18 → EfficientNet) | Un nouveau modèle dans le **registre** `models/`, choisi par la config. |
| Changer d'**interface** (web, API, ligne de commande) | Elles appellent toutes les **mêmes services**. |
| Changer la **règle médicale** (seuils plus prudents pour le LASIK) | Une seule classe : `RiskThresholds`, testée. |
| **Tester** sans GPU ni données | Le domaine et les services se testent avec de faux objets, en millisecondes. |

## 3. Principes de clean code appliqués

| Principe | Où dans le code |
|---|---|
| **S** — Responsabilité unique | `RiskThresholds` ne fait que classer un score ; `CornealMaps` ne fait que garantir les 4 cartes. |
| **O** — Ouvert/fermé | Ajouter un dataset = ajouter une classe, sans modifier les autres. |
| **L** — Substitution de Liskov | Tout adaptateur peut remplacer un autre : même interface. |
| **I** — Interfaces séparées | Petites interfaces ciblées (lire un dataset ≠ prédire ≠ expliquer). |
| **D** — Inversion des dépendances | Les services dépendent d'**interfaces**, pas de PyTorch directement. |
| Objets **immuables** (`frozen=True`) | Un examen ou un résultat ne peut pas être modifié par erreur après création. |
| **Validation à la construction** | Impossible de créer un examen sans ses 4 cartes ou un score hors de [0, 1]. |
| **Enum** plutôt que chaînes | `EyeSide.RIGHT` au lieu de `"OD"`, `"od"`, `"droit"`… |
| **Exceptions métier** | `MissingMapError` → message clair pour le médecin. |
| **Noms explicites** + typage strict | `mypy --strict` ; aucune fonction sans types. |

## 4. Outils de qualité

| Outil | Rôle | Quand |
|---|---|---|
| **Git** | historique des versions | toujours |
| **ruff** | lint (erreurs, style PEP 8, imports) + formatage | à chaque commit (pre-commit) et en CI |
| **mypy** (strict) | vérifie les types | à chaque commit et en CI |
| **pytest** + coverage | tests automatiques + mesure de couverture | à chaque commit et en CI |
| **pre-commit** | lance tout cela automatiquement avant chaque commit | `git commit` |
| **GitHub Actions** | refait toutes les vérifications sur un serveur neutre | à chaque push |
| **pyproject.toml** | un seul fichier pour dépendances + configuration des outils | — |

## 5. Structure « src layout »
Le code est dans `src/qarniya/` et non à la racine : on est obligé d'**installer** le paquet
(`pip install -e .`) pour le tester. Cela garantit que les tests utilisent le paquet tel qu'un utilisateur
l'installerait, et évite les imports « qui marchent par hasard ».

## 6. Journal des décisions (ADR — Architecture Decision Records)

| # | Décision | Pourquoi | Alternative écartée |
|---|---|---|---|
| 001 | Architecture en couches avec domaine pur | Évolutivité, testabilité, changer d'IA sans tout casser | Scripts monolithiques (le prototype) |
| 002 | Adaptateurs de datasets + config YAML | Exigence : changer de dataset sans modifier le code | Chemins et colonnes codés en dur |
| 003 | PyTorch + ResNet18 pré-entraîné | Petit dataset (~1 400 yeux), GPU 8 Go, Grad-CAM facile | Modèle entraîné de zéro (pas assez de données) |
| 004 | ruff + mypy strict + pytest + pre-commit + CI | Qualité automatique, pas seulement « promise » | Vérifications manuelles |
| 005 | FastAPI pour l'API | Documentation Swagger automatique, validation par types | Flask (moins de validation intégrée) |
| 006 | Données et modèles hors de Git | Taille + données de santé (loi 09-08) | Tout versionner |
| 007 | Prototype conservé dans `prototype/` | Démo de secours pendant la refonte | Le supprimer |
| 008 | Callbacks d'entraînement (checkpoint, CSV, TensorBoard) | Ajouter un suivi (MLflow…) sans toucher au `Trainer` | Tout coder dans la boucle |
| 009 | Préparation des images enregistrée DANS le modèle | Impossible d'utiliser le modèle avec une autre préparation | Configuration séparée, risque d'écart |
| 010 | Test évalué une seule fois (`--force` sinon), seuil choisi sur la validation | Pas de réglage involontaire sur le test | Ajuster le seuil sur le test |
| 011 | Détecteur d'images inhabituelles par plus proches voisins (empreintes) | Simple, sans ré-entraînement, mesurable (42/43 en simulation) | Faire confiance au score seul |
| 012 | Fusion images + chiffres **non** adoptée ; désaccord → « à surveiller » | La fusion manque un KC de plus au seuil prévu ; le désaccord est une information | Moyenne comme score principal |
| 013 | Interface en HTML/CSS/JS sans framework, servie par FastAPI | Aucune compilation, CSP stricte possible, une seule commande | React/Gradio (dépendances, surface d'attaque) |
| 014 | Rapport PDF dessiné avec matplotlib | Aucune nouvelle dépendance | reportlab / WeasyPrint |
| 015 | Docker PyTorch CPU, modèle monté en lecture seule | Tourne partout ; pas de données dans l'image ; changer de modèle sans reconstruire | Modèle copié dans l'image |
| 016 | OpenCV retiré, TensorBoard en extra `train` | Moins de dépendances et de surface d'attaque | Garder l'héritage du prototype |
| 017 | Le dossier `app/` prévu à l'étape 2 est supprimé : l'interface vit dans `api/static/` | Livrée avec le paquet Python et l'image Docker | Dossier séparé à installer à part |
