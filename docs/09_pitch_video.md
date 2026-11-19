# Étape 13 — Le pitch de 3 minutes et la vidéo

Deck : https://claude.ai/artifact/Vx8GcMNMXBbe8VLTowPzhy (le texte ci-dessous est aussi dans les notes de chaque slide).

## 1. Le texte, mot à mot (333 mots ≈ 2 min 35 à 130 mots/min)

Les ~25 secondes restantes sont la marge pour les gestes de la démo et les respirations. **Ne pas accélérer** :
un pitch trop rapide est le défaut n°1 des hackathons.

| Temps | Slide | Ce que vous dites |
|---|---|---|
| **0:00** | 1 · Couverture | Imaginez Yassine, dix-neuf ans. Sa vue baisse, ses lunettes ne suffisent plus. Le diagnostic tombe : kératocône avancé. Il aurait pu être freiné, s’il avait été détecté deux ans plus tôt. *(pause)* Nous sommes l’équipe NeuroLens, et voici **Qarniya AI**. |
| **0:18** | 2 · Problème | Le kératocône déforme la cornée des jeunes. Dans notre région, les études montrent **jusqu’à près de cinq pour cent** de la population touchée. Un traitement, le cross-linking, freine la maladie, mais il ne répare pas : **il faut agir tôt**. Or les formes débutantes sont discrètes, et lire ces cartes demande un expert. Avant un LASIK, un cas non vu peut coûter très cher. |
| **0:46** | 3 · Solution | Qarniya AI lit les quatre cartes de topographie d’un œil. En quelques secondes, il donne un niveau de risque, et surtout **il explique** : en orange, ce qui a fait monter le score, et le poids de chaque carte. Le médecin peut ajouter deux chiffres qu’il connaît déjà. Et si l’image est douteuse, l’outil sait dire : **je ne sais pas**. |
| **1:12** | 4 · Démo | Voici un vrai cas de nos données de test, un kératocône. Les images seules se trompent : **zéro pour cent**. Le médecin saisit le Kmax et l’épaisseur. Les chiffres disent **quatre-vingt-quatre pour cent**. L’outil voit le désaccord : l’œil n’est plus classé faible, mais **à surveiller**. Le médecin vérifie, et signe le rapport. *(pause)* Pour nous, **une IA médicale ne doit pas avoir toujours raison : elle doit savoir quand elle doute.** |
| **1:55** | 5 · Résultats | Nous l’avons testé sur deux cent soixante-sept yeux jamais vus, avec un protocole strict. Il détecte **quatre-vingt-quinze virgule six pour cent** des kératocônes, avec **une seule fausse alerte** sur cent soixante-seize yeux sains. Et nous publions nos limites : un seul appareil, pas encore de formes débutantes. |
| **2:18** | 6 · Confiance | Derrière la démo, un vrai logiciel : changer d’appareil tient dans un fichier de configuration, cent trente-huit tests automatiques, et un déploiement Docker sécurisé qui ne garde aucune image. |
| **2:38** | 7 · Impact | Nos premiers utilisateurs : les centres LASIK et le dépistage des jeunes. Prochaine étape : valider l’outil avec un ophtalmologue d’Oujda. *(pause)* **Qarniya AI : voir le kératocône avant qu’il ne vole la vue.** Merci. |

### Les 4 phrases à savoir par cœur (si tout le reste s’efface)
1. « Il aurait pu être freiné, s’il avait été détecté deux ans plus tôt. »
2. « Il faut agir tôt — et les formes débutantes sont discrètes. »
3. « Une IA médicale ne doit pas avoir toujours raison : elle doit savoir quand elle doute. »
4. « Qarniya AI : voir le kératocône avant qu’il ne vole la vue. »

### Précautions d’honnêteté (un jury le remarque)
- **Yassine est un exemple fictif** : si on vous le demande, dites-le tout de suite.
- **« Validation avec un ophtalmologue d’Oujda »** : dites « confirmée » seulement si le médecin a répondu oui, sinon « demande en cours ».
- Les chiffres viennent de **CornOrb (Algérie, Orbscan)**, pas de patients marocains : ne pas dire « testé au Maroc ».

## 2. La vidéo (si le jury la demande, et comme plan B)

### Scénario plan par plan
| Temps | À l’écran | Voix |
|---|---|---|
| 0:00 – 0:18 | Slide 1, puis vous en caméra (coin bas droit) | accroche Yassine |
| 0:18 – 0:46 | Slide 2 | problème |
| 0:46 – 1:12 | Slide 3, **zoom lent** sur la carte de chaleur orange | solution |
| 1:12 – 1:55 | **Capture d’écran réelle** de l’interface : glisser les 4 images G3H95 OD → 0 % → saisir 50.9 et 490 → bandeau « Désaccord » → clic « Rapport PDF » → le PDF s’ouvre | démo |
| 1:55 – 2:18 | Slide 5, apparition des chiffres | résultats |
| 2:18 – 2:38 | Slide 6, puis 3 secondes du **badge CI vert** sur GitHub | ingénierie |
| 2:38 – 3:00 | Slide 7, la phrase finale reste 3 secondes à l’écran | clôture |

### Enregistrer avec OBS Studio (gratuit)
1. **Paramètres → Sortie** : format **MP4**, encodeur matériel (NVENC si proposé), débit **8 000 Kbps**.
2. **Paramètres → Vidéo** : résolution **1920×1080**, **30 i/s**.
3. **Sources** : « Capture d’écran » (tout l’écran) + « Périphérique de capture vidéo » (webcam, réduite dans un coin) + micro.
4. Micro : **Filtres → Suppression du bruit** ; parler à 15 cm, dans une pièce calme.
5. Enregistrer **par morceaux** (une slide = un morceau) : une erreur ne fait recommencer qu’un morceau.

### Monter avec Clipchamp (inclus dans Windows) ou CapCut
- Coller les morceaux dans l’ordre, couper les silences de début et de fin.
- **Sous-titres automatiques** (le jury regarde parfois sans le son), relire les mots médicaux : kératocône, Kmax, cross-linking.
- Zoom sur les moments clés : carte de chaleur, bandeau « Désaccord », 95,6 %.
- Exporter en **1080p MP4**, vérifier : **durée ≤ 3:00**.

## 3. Le kit de démo
Le dossier `outputs/demo_kit/` (hors Git, il contient des images du dataset) est prêt :

| Dossier | Pour montrer | À saisir |
|---|---|---|
| `1_desaccord_G3H95_OD/` | **le cas phare** : images 0 %, chiffres 84 % → « à surveiller » | œil **OD**, Kmax **50.9**, épaisseur **490** |
| `2_keratocone_PN34D_OS/` | un kératocône net, chaleur sur le cône | œil **OS**, Kmax **47.9**, épaisseur **512** |
| `3_sain_63358_OS/` | un œil sain → « faible » | œil **OS** |
| `4_inhabituel_G3H95_OS/` | « ⚠ score non fiable » (kératocône très avancé, atypique) | œil **OS** |
| `5_refus_image_grise/` | « ✕ Analyse impossible » (une carte en noir et blanc) | œil **OD** |

Lancer l’application avant de présenter :
```powershell
qarniya serve outputs/runs/20261118_195456_cornorb_resnet18 --port 8765
```
