# Étape 1 — Cadrage du projet Qarniya AI

> **But de ce document :** dire *quel problème* on résout, *pour qui*, *ce qu'on construit (et ce qu'on ne construit pas)*,
> et *comment on saura que c'est réussi*. Toute décision technique des étapes suivantes doit pouvoir être justifiée par ce document.

---

## 1. Le problème

### 1.1 La maladie en 4 phrases
- Le **kératocône** : la cornée (la « vitre » à l'avant de l'œil) s'amincit et se déforme en **cône**.
- Il apparaît le plus souvent **à l'adolescence** et progresse jusqu'à 40–50 ans → il touche des **jeunes**, en pleine scolarité.
- Conséquence : vision floue et déformée, que les lunettes corrigent de plus en plus mal.
- Aux stades avancés : **greffe de cornée**.

### 1.2 Pourquoi le dépistage précoce est crucial
- Il existe un traitement qui **freine la progression** : le **cross-linking** (CXL, riboflavine + UV-A).
  Les essais randomisés montrent qu'il stoppe l'aggravation mesurée sur la topographie.
- **Mais il ne répare pas** une cornée déjà déformée : il faut donc le faire **tôt**.
- → Le problème n'est pas l'absence de traitement, c'est **la détection trop tardive**.

### 1.3 Pourquoi on le rate
1. **Les formes débutantes sont discrètes** : la vision peut être encore bonne, les lunettes corrigent.
2. **Lire les cartes de topographie demande de l'expertise** : 4 cartes colorées par œil, des indices à interpréter.
   Cette expertise est concentrée dans quelques centres spécialisés.
3. **Risque chirurgical** : un kératocône non détecté avant une chirurgie **LASIK** peut provoquer une **ectasie**
   (aggravation grave de la déformation) après l'opération. Les études rétrospectives montrent que des formes
   frustes non reconnues avant l'opération font partie des causes.

### 1.4 L'ampleur
| Donnée | Valeur | Source |
|---|---|---|
| Prévalence mondiale | **1,38 pour 1 000** personnes (méta-analyse 2019, > 50 millions de personnes, 15 pays) | Hashemi et al., Cornea 2020 (PubMed 31498247) |
| Moyen-Orient, Turquie, Inde | **1,4 à 4,8 %** selon les études (bien plus élevé) | revue Eastern Mediterranean, AVEH |
| Facteur régional évoqué | **consanguinité**, fréquente dans une ceinture « du Maroc au Pakistan » | revue Eastern Mediterranean |
| Maroc | **aucune étude de prévalence publiée trouvée** → c'est en soi un argument : on manque de données | — |

> ⚠️ À dire honnêtement au jury : « Il n'existe pas, à notre connaissance, de chiffre marocain. Les études régionales
> montrent des taux jusqu'à plusieurs pourcents, bien au-dessus de la moyenne mondiale. »

---

## 2. Pour qui ? (les utilisateurs)

| Persona | Situation | Ce dont il a besoin |
|---|---|---|
| **Dr Amina, ophtalmologue généraliste** (ville moyenne) | A un topographe, voit beaucoup de patients, n'est pas spécialiste de la cornée | Un **second avis rapide** et **expliqué** pour ne pas rater un cas débutant |
| **Dr Youssef, chirurgien réfractif** (centre LASIK) | Doit **exclure** tout kératocône avant d'opérer | Un **filet de sécurité** avant chaque opération |
| **Optométriste / campagne de dépistage** (écoles, jeunes allergiques qui se frottent les yeux) | Beaucoup d'examens, peu de temps | **Trier** rapidement qui doit voir un spécialiste en urgence |
| **Patient suivi** | Kératocône connu | Savoir **s'il s'aggrave** → décider du cross-linking à temps |

**Qui n'est PAS l'utilisateur :** le patient seul, sans médecin. L'outil ne remplace jamais l'ophtalmologue.

---

## 3. La solution

> **Qarniya AI lit les 4 cartes de topographie d'un œil et donne en quelques secondes :**
> 1. un **score de risque** de kératocône (0–100 %) et un niveau (faible / à surveiller / élevé) ;
> 2. une **carte de chaleur** qui montre **où** se trouve la zone suspecte ;
> 3. l'**évolution** entre deux examens du même œil (progression → penser au cross-linking) ;
> 4. un **refus explicite** (« image non exploitable ») quand l'entrée est douteuse.

### Cas d'usage (user stories)
| ID | En tant que… | je veux… | afin de… | Priorité |
|---|---|---|---|---|
| US1 | ophtalmologue | déposer les 4 cartes d'un œil et obtenir un score + carte de chaleur | ne pas rater un kératocône débutant | **Indispensable** |
| US2 | ophtalmologue | comparer deux examens du même œil | détecter une progression et traiter à temps | **Indispensable** |
| US3 | ophtalmologue | que l'outil refuse une image floue / incomplète / d'un autre appareil | ne pas recevoir un faux résultat | **Indispensable** |
| US4 | chirurgien réfractif | analyser un dossier entier de patients et les trier par risque | sécuriser mon planning opératoire | Important |
| US5 | ophtalmologue | télécharger un rapport PDF | l'ajouter au dossier patient | Bonus |
| US6 | ingénieur / chercheur | brancher un nouveau dataset ou un nouvel appareil par configuration | faire évoluer l'outil sans réécrire le code | **Indispensable** (exigence technique) |

---

## 4. Périmètre

| ✅ Dans le projet (hackathon) | ❌ Hors projet |
|---|---|
| Classification **normal / kératocône** sur cartes Orbscan (dataset CornOrb) | Poser un **diagnostic** (le médecin décide) |
| Carte de chaleur explicative (Grad-CAM) | Recommander un **traitement** précis |
| Comparaison de deux examens | Certification comme **dispositif médical** |
| Contrôle qualité des entrées + refus | Stockage de données patients identifiantes |
| Architecture ouverte à d'autres datasets / appareils | Entraînement sur Pentacam / Sirius (pas de données publiques suffisantes pour l'instant) |
| API + interface web + documentation | Application mobile |

---

## 5. Indicateurs de réussite (KPI)

### Médicaux (mesurés sur le jeu de **test**, jamais vu à l'entraînement)
| Indicateur | Cible | Pourquoi |
|---|---|---|
| **Sensibilité** (kératocônes détectés) | **≥ 95 %** | rater un KC = le pire cas (pas de traitement, ou LASIK dangereux) |
| **Spécificité** (normaux reconnus) | ≥ 90 % | éviter d'inquiéter et de surcharger les spécialistes |
| **AUC** | ≥ 0,95 | qualité globale du classement, indépendante du seuil |
| Supérieur à la **baseline** chiffres seuls (Kmax, épaisseur…) | oui | prouver que l'image apporte quelque chose |

> Résultat actuel du prototype : AUC 0,998 · sensibilité 95,6 % · spécificité 98,9 % (baseline : sensibilité 81,3 %).

### Produit / technique
| Indicateur | Cible |
|---|---|
| Temps d'analyse d'un œil | < 3 secondes |
| Ajouter un nouveau dataset | **1 fichier de configuration + 1 adaptateur**, 0 modification du reste du code |
| Couverture de tests du code métier | ≥ 70 % |
| Reproductibilité | même config + même graine → même résultat |
| Validation clinique | avis qualitatif d'au moins **1 ophtalmologue** sur des cas réels |

---

## 6. Contraintes

| Contrainte | Conséquence pour le projet |
|---|---|
| **Loi 09-08** (protection des données personnelles, contrôle par la **CNDP**) | Données anonymisées (code aléatoire), aucune identité, suppression sur demande, accord de l'établissement |
| **Pas un dispositif médical certifié** | Message affiché partout : « outil d'aide, pas de diagnostic » |
| **48 h de hackathon**, équipe réduite | Priorités strictes (indispensable / important / bonus) |
| **Matériel** : 1 GPU 8 Go (Quadro RTX 4000) ou Colab T4 | Modèles légers (ResNet18), images réduites à 256 px |
| **Un seul dataset public** adapté (CornOrb, Orbscan 3, Algérie) | Architecture prête pour d'autres sources ; limites annoncées |

---

## 7. Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Le modèle lit le **texte imprimé** (chiffres) au lieu de la forme | Résultats trompeurs | Vérifier avec Grad-CAM ; option : masquer le texte |
| **Pas de classe « suspect »** dans CornOrb | On ne valide pas les formes débutantes | Le dire ; seuil prudent ; données du médecin |
| **Changement d'appareil** (Pentacam, Sirius) | Performance qui chute | Contrôle qualité qui **refuse** ; ré-entraînement via un nouvel adaptateur |
| Population **algérienne** ≠ marocaine | Biais possible | Proche géographiquement ; validation locale avec le médecin |
| **Surconfiance** du médecin dans l'outil | Erreur médicale | Score + carte de chaleur + avertissement : il garde le contrôle |
| Démo qui plante le jour J | Mauvaise impression | **Vidéo de secours** de la démo |

---

## 8. Hypothèses à vérifier
1. Les ophtalmologues trouvent la carte de chaleur **utile et cohérente** → à valider avec notre médecin.
2. Un score sur les cartes seules suffit pour un **premier tri** (pas besoin de tous les indices de l'appareil).
3. La comparaison de deux examens par le score reflète la **progression clinique** → nécessite des données répétées (médecin).

---

## Sources
- Prévalence mondiale : Hashemi H. et al., *The Prevalence and Risk Factors for Keratoconus: A Systematic Review and Meta-Analysis*, Cornea 2020 — https://www.pubmed.gov/31498247/
- Région Méditerranée orientale : *Keratoconus in Eastern Mediterranean Region: Prevalence and risk factors*, AVEH — https://avehjournal.org/index.php/aveh/article/view/1044/2899
- Cross-linking, essais randomisés : HTA suisse — https://www.bag.admin.ch/dam/en/sd-web/MhVvmyQckz1a/h0054uvcl-hta-shortreport.pdf ; Cochrane — https://www.cochrane.org/CD010621/EYES_corneal-collagen-cross-linking-for-keratoconus
- Cross-linking, 10 ans : Raiskup F. et al., J Cataract Refract Surg 2015 — https://pubmed.ncbi.nlm.nih.gov/25532633/
- Ectasie post-LASIK et formes frustes : Moshirfar et al., Clinical Ophthalmology — https://www.dovepress.com/rate-of-ectasia-and-incidence-of-irregular-topography-in-patients-with-peer-reviewed-fulltext-article-OPTH
- Dataset : Lazouni M.E.A. et al., *CornOrb*, Zenodo 2025, DOI 10.5281/zenodo.17121301 (CC BY 4.0)
