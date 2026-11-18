# Étape 7 — L'API et l'interface du médecin

## 1. Lancer l'application

```powershell
qarniya prepare outputs/runs/20261118_195456_cornorb_resnet18
```
(une seule fois : construit le détecteur d'images inhabituelles et l'avis des chiffres)

```powershell
qarniya serve outputs/runs/20261118_195456_cornorb_resnet18 --port 8765
```
Puis ouvrir **http://127.0.0.1:8765** · documentation interactive de l'API : **http://127.0.0.1:8765/docs**

## 2. Ce que voit le médecin

**Onglet « Analyse d'un œil »**
1. Il glisse **les 4 images d'un coup** : elles se rangent seules d'après leur nom (axiale / antérieure / postérieure / pachymétrie, en français ou en anglais).
2. Il choisit l'œil (OD / OS) et, s'il le souhaite, saisit **Kmax et épaisseur minimale**.
3. Il obtient : le **score**, le **niveau** (icône + couleur + texte), une **jauge** 0–30–70–100 %, une **phrase** d'explication,
   les **4 cartes avec la chaleur** et la **part de chaque carte** dans la décision, et l'**avis des chiffres** à côté.

**Bandeaux d'alerte** (jamais la couleur seule : toujours une icône et un texte)
| Bandeau | Quand | Effet |
|---|---|---|
| ✕ Analyse impossible | image illisible, vide, sans couleurs, mauvais format | **aucun score** |
| ⚠ Score non fiable | image inhabituelle (autre appareil ?) | score **grisé et barré** |
| ◐ Désaccord images / chiffres | l'un dit « faible », l'autre « élevé » | niveau relevé à **« à surveiller »** |

**Onglet « Suivi de progression »** : examen ancien + examen récent → « Score en hausse nette (12 % → 58 %) : possible
progression, à contrôler » / « stable » / « en baisse ». Si l'un des examens n'est pas fiable : **comparaison refusée**.

## 3. Le cas qui montre tout l'intérêt du deuxième avis

**G3H95 OD** est le kératocône que le modèle images avait manqué à l'étape 5 (score 0 %).
Avec Kmax 50,9 et épaisseur 490 saisis par le médecin :

> *Risque **à surveiller** (images : 0 %). Avis des chiffres : élevé (84 %). **Désaccord images / chiffres : à examiner.***

→ Le kératocône manqué n'est plus classé « faible » : **le filet de sécurité fonctionne**.
(Les deux autres cas testés, un kératocône et un œil sain, donnent des avis concordants.)

## 4. L'API (pour d'autres logiciels)

| Méthode | Adresse | Rôle |
|---|---|---|
| GET | `/api/health` | état : modèle, détecteur, avis des chiffres |
| POST | `/api/analyses` | `eye` + `axial`, `anterior`, `posterior`, `pachymetry` (+ `kmax`, `pachymetry_thinnest`) |
| POST | `/api/progressions` | `eye` + `before_*` (4 cartes) + `after_*` (4 cartes) |
| GET | `/docs` | documentation Swagger générée automatiquement |

Réponse : JSON validé (statut, score, niveaux, résumé, raisons, avis des chiffres, désaccord, 4 images de chaleur en PNG base64).

## 5. Sécurité et confidentialité

| Risque | Mesure |
|---|---|
| Faux fichier (script déguisé en .png) | l'image est **décodée et vérifiée** (`verify`) : le nom et l'extension ne comptent pas → erreur 415 |
| Fichier énorme / « bombe » de décompression | 10 Mo et 16 mégapixels maximum par carte → erreur 413 |
| Nom de fichier malveillant (`../../`) | le nom envoyé n'est **jamais** utilisé comme chemin |
| Données de santé conservées | dossier temporaire supprimé à la fin de chaque requête, images **ré-encodées** (métadonnées supprimées), rien n'est journalisé |
| Saisie aberrante (Kmax 4,75 au lieu de 47,5) | refusée avec un message clair (valeurs plausibles) |
| Injection dans la page | tout texte du serveur inséré avec `textContent`, **CSP stricte** (`default-src 'self'`), JavaScript dans un fichier séparé |
| Clic piégé (iframe) | `X-Frame-Options: DENY`, `frame-ancestors 'none'` |
| Accès depuis le réseau | écoute sur **127.0.0.1** par défaut ; avertissement si on ouvre au réseau (pas d'authentification) |
| Deux requêtes simultanées | un verrou : un seul calcul à la fois (le modèle et Grad-CAM partagent un état) |

## 6. Le code

| Fichier | Rôle |
|---|---|
| `domain/risk.py` | `OpinionComparison` : la règle « désaccord → à surveiller » (domaine pur, testé) |
| `services/clinical.py` | avis des chiffres (Kmax + épaisseur), sauvegarde JSON, garde-fou de saisie |
| `services/progression.py` | comparaison de deux examens, refus si l'un n'est pas fiable |
| `api/app.py` | API FastAPI, contrôle des fichiers, en-têtes de sécurité |
| `api/schemas.py` | format des réponses (pydantic) |
| `api/static/` | interface web : `index.html`, `style.css`, `app.js` (sans framework ni compilation) |
| `cli.py` | nouvelles commandes `prepare` et `serve` |

Tests : **125** au total, dont l'API de bout en bout (analyse, progression, en-têtes, faux PNG, GIF, fichier trop gros,
carte manquante, œil invalide, Kmax aberrant) et l'avis des chiffres (JSON, valeurs impossibles).
Vérifié aussi dans un vrai navigateur : affichage, bandeaux, score grisé, onglet progression, aucune erreur de CSP.
