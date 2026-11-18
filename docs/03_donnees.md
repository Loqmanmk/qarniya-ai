# Étape 3 — La couche données

## 1. Le problème à résoudre
Chaque source de données a son propre format :

| Source | Fichiers | Œil gauche | Cartes | Tableau |
|---|---|---|---|---|
| CornOrb | `9C6SM/OD/9C6SM_OD_Axial.png` | `OS` | `Axial`, `Anterior`… | CSV anglais, label `0/1` |
| Notre médecin | `P017_OG_axiale.png` | `OG` | `axiale`, `anterieure`… | CSV français, `normal/suspect/kératocône` |
| Nouveaux patients | dossier d'images, **pas de tableau** | `OG` | idem | aucun diagnostic |

**Mauvaise solution** (le prototype) : écrire ces détails en dur dans le code → chaque nouveau dataset = modifier le code.
**Notre solution** : le patron **Adaptateur** + une **configuration YAML**.

## 2. Comment ça marche

```
 configs/data/cornorb.yaml ──► DatasetConfig (pydantic : validé, typé)
                                     │
                                     ▼  adapter: csv_images
                              registre d'adaptateurs ──► CsvImagesAdapter / ImageFolderAdapter / (futur)
                                     │
                                     ▼  .load()
                              LoadResult
                               ├─ exams    : tuple[CornealExam]   ← objets du DOMAINE, format d'origine oublié
                               ├─ skipped  : chaque exclusion + sa raison (traçabilité)
                               └─ predefined_splits
                                     │
                                     ▼  split_dataset()
                              DatasetSplits(train, val, test)  ← découpage PAR PATIENT, contrôle de fuite
                                     │
                                     ▼
                              ImageQualityChecker  ← refuse les images vides, minuscules, illisibles
```

## 3. Les fichiers

| Fichier | Rôle |
|---|---|
| `data/config.py` | `DatasetConfig` : description d'un dataset, **validée** (faute de frappe = erreur claire) |
| `data/adapters/base.py` | l'**interface** `DatasetAdapter` (une seule méthode : `load()`) |
| `data/adapters/registry.py` | le **registre** : nom dans le YAML → classe Python |
| `data/adapters/csv_images.py` | adaptateur « tableau + images » (CornOrb, médecin) |
| `data/adapters/image_folder.py` | adaptateur « dossier d'images seul » (nouveaux patients) |
| `data/adapters/paths.py` | construit et **reconnaît** les noms de fichiers à partir du motif |
| `data/splitting.py` | découpage train / val / test **par patient** |
| `data/validation.py` | contrôle qualité des images |
| `configs/data/*.yaml` | les 3 datasets déjà décrits |

## 4. Ajouter un nouveau dataset

**Cas 1 — même genre de format (tableau + images, ou dossier d'images)** : copier un YAML, changer les libellés.
```powershell
qarniya data inspect configs/data/mon_dataset.yaml --check-images
```

**Cas 2 — format vraiment différent** (ex. export XML d'un Pentacam) : écrire une classe.
```python
@register_adapter("pentacam_xml")
class PentacamXmlAdapter(DatasetAdapter):
    def load(self) -> LoadResult: ...  # lire le XML, renvoyer des CornealExam
```
Puis `adapter: pentacam_xml` dans le YAML. **Aucun autre fichier n'est modifié.**
(Le test `test_new_adapter_can_be_plugged_without_touching_existing_code` le démontre.)

## 5. Ce que la couche données a trouvé dans CornOrb

```
qarniya data inspect configs/data/cornorb.yaml --check-images
```
| Constat | Traitement |
|---|---|
| 1 454 lignes dans le CSV | — |
| **93 lignes en double** (46 yeux, même patient + même œil, valeurs différentes, un seul jeu d'images) | écartées : impossible de savoir quelle ligne correspond aux images |
| Code patient **`3E+132`** dans le CSV, dossier **`3E132`** | bug classique d'**Excel** (notation scientifique) → corrigé **dans le YAML** (`patient_code_fixes`), 2 yeux récupérés |
| **1 361 yeux** utilisables | train 872 · val 222 · test 267 |
| Contrôle de fuite | ✅ aucun patient dans deux ensembles |
| Contrôle qualité | ✅ 0 image refusée |

## 6. Le découpage par patient (point clé pour le jury)
Les deux yeux d'un même patient se ressemblent (même génétique, souvent même stade).
Si l'œil droit est en entraînement et l'œil gauche en test, le modèle est testé sur un patient « déjà vu »
→ **score artificiellement gonflé** : c'est une **fuite de données** (*data leakage*).
→ On découpe **par patient**, et `leaked_patients()` le **vérifie** à chaque chargement.

On **stratifie** aussi : chaque ensemble garde à peu près la même proportion de malades.

## 7. Tests ajoutés (60 au total, couverture 94 %)
- configuration : YAML valides acceptés, 6 types d'erreurs refusés avec un message clair, YAML du projet toujours valides ;
- adaptateurs : traduction fidèle, images manquantes, doublons, étiquette inconnue, correction de code, colonnes absentes,
  noms de fichiers du médecin (`OG`, codes avec `_`), fichiers non reconnus, adaptateur inconnu, **ajout d'un adaptateur à chaud** ;
- découpage : **aucune fuite**, proportions, reproductibilité (même graine → même découpage) ;
- qualité d'image : image vide, minuscule, panoramique, fichier corrompu ;
- intégration : le vrai CornOrb donne exactement 1 361 / 872 / 222 / 267 (ignoré s'il n'est pas téléchargé).
