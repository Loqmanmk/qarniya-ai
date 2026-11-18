# Fiche du modèle — Qarniya AI `cornorb_resnet18`

> Format inspiré des *Model Cards* (Mitchell et al., 2019), standard de transparence en IA.
> **En une phrase :** un modèle qui estime, à partir des 4 cartes de topographie d'un œil (Orbscan), la probabilité
> d'un kératocône, pour **aider** l'ophtalmologue à ne pas en manquer. Il ne pose pas de diagnostic.

---

## 1. Identité

| | |
|---|---|
| Nom | `cornorb_resnet18` (Qarniya AI 0.1.0) |
| Équipe | NeuroLens — Hackathon Nexus HealthTech 2026, EHEIM Oujda |
| Date | entraîné le 18/11/2026, commit `17674fd` |
| Type | classification binaire : kératocône (ou suspect) / non |
| Architecture | ResNet18 pré-entraîné ImageNet, **partagé** entre les 4 cartes, + tête de fusion (2048 → 128 → 1) |
| Entrées | 4 cartes PNG/JPEG d'un œil : courbure axiale, élévation antérieure, élévation postérieure, pachymétrie ; œil gauche retourné en miroir ; 256 × 256 |
| Sortie | probabilité 0–1 → niveau **faible** (< 30 %), **à surveiller** (30–70 %), **élevé** (≥ 70 %) |
| Fichiers | `model.pt` (poids + préparation des images + métadonnées), `novelty.npz`, `clinical_opinion.json` |
| Licence des données | CornOrb, CC BY 4.0 |

## 2. Usage prévu

| ✅ Prévu | ❌ Hors usage |
|---|---|
| **Second avis** pour un ophtalmologue qui lit des cartes Orbscan | Diagnostic autonome, sans médecin |
| Repérer les yeux à revoir en **priorité** (dépistage, avant LASIK) | Décider seul d'un traitement (cross-linking, greffe) ou d'une contre-indication chirurgicale |
| Suivre l'évolution du score entre deux examens | Autres appareils (Pentacam, Sirius…) **sans ré-entraînement** |
| Recherche, enseignement, démonstration | Usage par le patient lui-même ; tout usage comme dispositif médical (**non certifié**) |

**Utilisateurs :** ophtalmologues, chirurgiens réfractifs, équipes de dépistage encadrées par un médecin.

## 3. Données d'entraînement et d'évaluation

**Source :** CornOrb (Lazouni et al., Zenodo 2025, DOI 10.5281/zenodo.17121301) — Orbscan 3, centres d'ophtalmologie
algériens. 1 454 lignes, 744 patients.

| | Valeur |
|---|---|
| Âge | 13 à 65 ans (médiane 30) |
| Sexe | 947 femmes (65 %), 507 hommes (35 %) |
| Classes | 889 normaux, 565 kératocônes — **aucun cas « suspect »** |
| Sévérité des kératocônes | Kmax médian 52,1 D ; seulement **23 %** ont un Kmax < 48 D (formes peu avancées) |
| Exclusions | 93 lignes en double ambiguës (46 yeux) ; 1 code patient corrigé (`3E+132` → `3E132`, bug Excel) |
| Utilisés | **1 361 yeux** |
| Découpage | celui du dataset, **par patient** (aucun patient dans deux ensembles, vérifié) : entraînement 872 · validation 222 · **test 267** (143 patients) |

## 4. Performances (jeu de test, jamais vu)

Seuil choisi **sur la validation** pour viser 95 % de sensibilité (seuil = 0,36), appliqué tel quel au test.
Fourchettes à 95 % par bootstrap **par patient** (2 000 tirages). Test évalué une seule fois.

| Modèle | AUC | Sensibilité | Spécificité | KC manqués | Fausses alertes |
|---|---|---|---|---|---|
| **Images (ce modèle)** | **0,996** [0,987–1,000] | **95,6 %** [90,9–98,9] | **99,4 %** [98,2–100] | 4 / 91 | 1 / 176 |
| Chiffres seuls (Kmax, épaisseurs, astigmatisme, asphéricités) | 0,983 [0,957–0,999] | 94,5 % [89,3–98,8] | 97,7 % [94,6–100] | 5 / 91 | 4 / 176 |
| Fusion (vote moyen) | 1,000 [0,998–1,000] | 94,5 % [89,4–98,8] | 99,4 % [98,2–100] | 5 / 91 | 1 / 176 |

**À lire avec prudence :**
- les fourchettes se **chevauchent** : la supériorité des images sur les chiffres n'est pas démontrée statistiquement ;
- les erreurs des images et des chiffres sont **complémentaires** (3 des 4 KC manqués par les images sont vus par les
  chiffres) → dans l'outil, un **désaccord** fait passer l'œil en « à surveiller » ;
- la fusion n'est **pas** adoptée comme score principal (seuil mal placé, voir `docs/05_evaluation.md` § 4).

## 5. Garde-fous intégrés

| Garde-fou | Comportement | Mesure |
|---|---|---|
| Contrôle qualité | refuse une carte illisible, vide, minuscule, sans couleurs → **aucun score** | 0 carte CornOrb refusée à tort |
| Détecteur d'images inhabituelles | « score non fiable » si l'œil ne ressemble pas aux yeux d'entraînement | 1,9 % des vrais yeux de test signalés ; **42/43** erreurs provoquées par un autre appareil **simulé** signalées |
| Deuxième avis des chiffres | désaccord franc → niveau « à surveiller » | rattrape G3H95 OD, KC manqué par les images |
| Carte de chaleur + part de chaque carte | montre ce qui a fait monter le score | qualitatif ; grille 8 × 8, donc floue |

## 6. Limites connues

1. **Un seul appareil (Orbscan 3)** : sur Pentacam ou Sirius, l'échelle de couleurs diffère. Le détecteur d'images
   inhabituelles le signale (en simulation), mais le modèle doit être **ré-entraîné** pour ces appareils.
2. **Pas de formes suspectes / frustes** dans les données : ce sont les cas les plus difficiles et les plus utiles au
   dépistage. **Les performances sur ces cas sont inconnues.** Le jeu de test contient surtout des cas nets
   (77 % des KC ont un Kmax ≥ 48 D) : les chiffres de la section 4 sont probablement **optimistes** pour un dépistage réel.
3. **Une seule population** (Algérie, 65 % de femmes, 13–65 ans) : proche du Maroc, mais à valider localement ; pas de
   données d'enfants de moins de 13 ans.
4. **Petit jeu de test** (267 yeux) : d'où les fourchettes de ± 4 à 5 points sur la sensibilité.
5. **Progression** : aucune donnée longitudinale ; la comparaison de deux examens n'a **pas été validée**
   cliniquement (seuil de variation de 15 points fixé a priori).
6. **Calibration non mesurée** : un score de 80 % n'est pas garanti signifier « 8 cas sur 10 ». Il sert à **classer**,
   les niveaux sont des repères.
7. **Carte de chaleur** : approximative (résolution 8 × 8), peut déborder de la cornée ; ce n'est pas une preuve.
8. **Étiquettes** : un cas étiqueté « normal » (66NY2 OS) présente une asymétrie inférieure qui mériterait l'avis d'un
   expert : les étiquettes du dataset peuvent elles-mêmes contenir des erreurs.

## 7. Considérations éthiques et réglementaires

- **Pas un dispositif médical certifié** : message affiché dans l'interface, l'API et chaque rapport PDF.
- **Loi 09-08 (CNDP)** : aucune identité traitée ; référence anonyme seulement dans les rapports ; images supprimées
  après chaque analyse ; données jamais versionnées dans Git.
- **Risque de surconfiance** : atténué par les niveaux (pas de « oui/non »), la carte de chaleur, l'avis des chiffres,
  les alertes « non fiable » et la zone de signature du médecin dans le rapport.
- **Équité** : performances par sexe et par âge **non encore mesurées** (prochaine vérification).

## 8. Comment reproduire

```powershell
qarniya data inspect configs/data/cornorb.yaml --check-images
qarniya train configs/experiments/cornorb_resnet18.yaml
qarniya evaluate outputs/runs/<run>
qarniya prepare outputs/runs/<run>
```
Graine fixe (42), configuration copiée dans chaque dossier de résultats, commit Git enregistré dans le modèle.

## 9. Prochaines étapes

1. Évaluer sur les **données de l'ophtalmologue partenaire** (Oujda) : nouvelle population, cas suspects, examens répétés.
2. Mesurer la **calibration** et les performances **par sous-groupe** (sexe, âge, sévérité).
3. Ré-entraîner pour **Pentacam / Sirius** via un nouvel adaptateur de données.
4. Mettre à jour **PyTorch** (avis de sécurité, voir `SECURITY.md`) puis ré-entraîner et ré-évaluer.
