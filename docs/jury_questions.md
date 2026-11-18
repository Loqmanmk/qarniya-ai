# Questions probables du jury — et quoi répondre

> Ce fichier grandira à chaque étape. Apprenez l'**idée**, pas le texte par cœur.
> Règle d'or : si vous ne savez pas, dites « Bonne question, nous ne l'avons pas encore mesuré, c'est dans nos prochaines étapes ».
> Un jury respecte l'honnêteté bien plus qu'une réponse inventée.

---

## Étape 1 — Le problème et le besoin

**Q1. Pourquoi le kératocône ? Ce n'est pas une maladie rare ?**
> Elle touche des **jeunes** (début à l'adolescence), elle est **plus fréquente dans notre région** que dans le monde
> (études du Moyen-Orient : 1,4 à 4,8 % contre 0,14 % au niveau mondial), et surtout il existe un traitement efficace
> **seulement si on la détecte tôt**. C'est le cas idéal pour l'IA : un problème de **détection**, pas de traitement.

**Q2. Quel est le chiffre au Maroc ?**
> À notre connaissance, aucune étude de prévalence marocaine n'est publiée. C'est justement un manque :
> un outil comme le nôtre, utilisé en dépistage, pourrait aussi **produire ces données**.

**Q3. Les médecins savent déjà lire les cartes. Pourquoi une IA ?**
> Les spécialistes de la cornée, oui. Mais ils sont peu nombreux et concentrés dans les grandes villes.
> L'ophtalmologue généraliste voit beaucoup de patients et les **formes débutantes sont subtiles**.
> Nous apportons un **second avis instantané et expliqué**, pas un remplacement.

**Q4. Votre outil pose un diagnostic ?**
> Non. Il donne un **score de risque** et montre **où** il voit une anomalie. La décision reste celle du médecin.
> C'est affiché dans l'interface et dans le rapport.

**Q5. Qui va payer / utiliser ça ?**
> Premier client naturel : les **centres de chirurgie réfractive (LASIK)** — un kératocône non détecté avant
> l'opération est leur pire risque (ectasie). Ensuite : cabinets d'ophtalmologie et campagnes de dépistage.

**Q6. Pourquoi la sensibilité est-elle votre indicateur principal, et pas la précision (accuracy) ?**
> Parce que les deux erreurs n'ont pas le même coût. **Rater un kératocône** (faux négatif) = pas de traitement à temps,
> ou un LASIK dangereux. **Une fausse alerte** (faux positif) = un examen de contrôle en plus. On préfère donc
> détecter presque tous les malades (sensibilité ≥ 95 %), quitte à quelques fausses alertes.

**Q7. Et les données personnelles ?**
> Nous respectons la **loi 09-08** : données anonymisées par un code aléatoire, aucune identité, utilisation limitée
> au projet, suppression sur demande, avec l'accord de l'établissement. Le dataset public CornOrb est déjà anonymisé.

**Q8. Qu'est-ce qui vous différencie des logiciels intégrés aux appareils (indices Pentacam, etc.) ?**
> 1) Notre outil est **indépendant de l'appareil** (architecture par adaptateurs) ; 2) il **montre la zone** suspecte ;
> 3) il **suit la progression** entre deux examens ; 4) il **refuse** les entrées douteuses au lieu de répondre au hasard.
> (Ne dites pas que nous sommes « meilleurs » que ces indices : nous ne les avons pas comparés.)

**Q9. Quelles sont les limites de votre projet ?**
> Un seul dataset public (Orbscan, Algérie), deux classes seulement (pas de « suspect »), pas encore de validation
> clinique sur des patients marocains. C'est pourquoi nous collaborons avec un ophtalmologue d'Oujda.

---

## Étape 2 — Architecture et qualité du code

**Q10. Pourquoi une architecture aussi structurée pour un hackathon ?**
> Parce que notre objectif n'est pas une démo jetable mais un outil qui **survit au hackathon** : changer de dataset,
> d'appareil ou de modèle sans tout réécrire. Le prototype du premier jour était un ensemble de scripts ; il marchait,
> mais changer de dataset demandait de modifier 4 fichiers. Maintenant c'est un adaptateur et un fichier YAML.

**Q11. Comment garantissez-vous que l'architecture est respectée ?**
> Par un **test automatique** (`test_architecture.py`) : il lit le code du domaine et échoue si quelqu'un y importe
> PyTorch, pandas ou une autre bibliothèque technique. La règle n'est pas une promesse, elle est vérifiée à chaque commit.

**Q12. Qu'est-ce que le « domaine » ?**
> Le vocabulaire médical du projet en pur Python : un examen, ses 4 cartes, un score de risque, une progression.
> Il ne sait pas qu'il existe une IA. Si demain on remplace le réseau de neurones, le domaine ne change pas.

**Q13. Pourquoi des objets immuables (`frozen=True`) ?**
> Un résultat médical ne doit pas pouvoir être modifié par erreur après avoir été calculé. Et la validation se fait
> à la création : impossible de créer un examen sans ses 4 cartes, ou un score supérieur à 100 %.

**Q14. Que vérifient ruff, mypy, pytest, pre-commit, la CI ?**
> ruff : style et erreurs fréquentes · mypy : cohérence des types (mode strict) · pytest : le comportement ·
> pre-commit : lance tout ça **avant chaque commit** · CI GitHub : refait tout sur un serveur neutre à chaque push.
> Aujourd'hui : 22 tests, 97 % de couverture du code existant, 0 erreur de lint ou de type.

**Q15. Qu'est-ce que SOLID ? Donnez un exemple dans votre code.**
> Cinq principes de conception orientée objet. Exemple du « O » (ouvert/fermé) : pour ajouter le dataset du médecin,
> on **ajoute** une classe adaptateur, on ne **modifie** aucune classe existante. Exemple du « S » : `RiskThresholds`
> ne fait qu'une chose, transformer un score en niveau de risque.

**Q16. Pourquoi les seuils de risque sont-ils configurables ?**
> Parce que le besoin médical change selon le contexte : un centre LASIK veut être **plus prudent** (seuils plus bas,
> plus de fausses alertes mais moins de kératocônes manqués) qu'une campagne de dépistage de masse.

---

## Étape 3 — Les données

**Q17. Comment changez-vous de dataset ?**
> Avec un fichier YAML qui décrit le format : où sont les images, comment s'appellent les colonnes, comment est noté
> l'œil gauche (`OS` ou `OG`)… Si le format est vraiment nouveau (ex. XML d'un appareil), on écrit **une** classe
> adaptateur et on l'enregistre ; aucun autre fichier ne change. Démonstration : `qarniya data inspect configs/data/medecin.yaml`.

**Q18. Qu'est-ce que le patron Adaptateur ?**
> Une classe qui traduit un format externe vers le format interne. Ici, chaque adaptateur transforme sa source en
> `CornealExam` (objet du domaine). Le modèle et l'interface ne voient jamais le format d'origine.

**Q19. Pourquoi découper par patient et pas par œil ?**
> Les deux yeux d'un patient se ressemblent. Si l'un est en entraînement et l'autre en test, le modèle est évalué sur
> un patient déjà vu : le score est gonflé, c'est une **fuite de données**. Notre code découpe par patient et
> **vérifie automatiquement** qu'aucun patient n'est dans deux ensembles.

**Q20. Avez-vous trouvé des problèmes dans les données ?**
> Oui, deux. 1) 46 yeux apparaissent deux fois dans le CSV avec des valeurs différentes mais un seul jeu d'images :
> on ne peut pas savoir quelle ligne est la bonne, on les écarte (93 lignes). 2) Un code patient `3E132` transformé
> en `3E+132` par Excel : corrigé dans la configuration, pas dans le code. Chaque exclusion est tracée avec sa raison.

**Q21. Pourquoi ne pas simplement supprimer les lignes en double ?**
> Garder une des deux au hasard, c'est risquer d'associer de mauvaises valeurs (Kmax, diagnostic) aux images.
> En médecine, mieux vaut **moins de données sûres** que plus de données douteuses.

**Q22. Que se passe-t-il si le médecin envoie une image floue ou incomplète ?**
> Le contrôle qualité la refuse avec un message précis (« pachymétrie : image vide »), et un œil sans ses 4 cartes
> n'est pas analysé. L'outil dit « je ne peux pas » au lieu de donner un score au hasard.

**Q23. Comment être sûr que tout cela marche ?**
> 60 tests automatiques sur de faux mini-datasets générés à la volée (ils tournent aussi sur GitHub, sans données
> réelles), plus un test sur le vrai CornOrb qui vérifie les chiffres exacts. Couverture : 94 %.

---

## Étape 4 — Le modèle et l'entraînement

**Q24. Quel modèle utilisez-vous et pourquoi ?**
> Un ResNet18 pré-entraîné sur ImageNet, partagé entre les 4 cartes, avec une petite tête qui fusionne les 4
> descripteurs. Pré-entraîné : il sait déjà voir formes et contours, on l'adapte (*transfert d'apprentissage*).
> Partagé : 4 fois moins de paramètres, essentiel avec moins de 1 000 yeux d'entraînement.

**Q25. Pourquoi pas un modèle plus gros (Vision Transformer, etc.) ?**
> Avec ~900 yeux d'entraînement, un gros modèle surapprend (il mémorise au lieu de généraliser). Et ResNet18 tourne
> en temps réel sur un PC ordinaire, ce qui compte pour un cabinet. Notre architecture permet d'en essayer d'autres
> par configuration (EfficientNet est déjà disponible).

**Q26. Qu'est-ce que le surapprentissage et comment l'évitez-vous ?**
> Le modèle apprend par cœur les images d'entraînement et échoue sur de nouvelles. Parades : modèle pré-entraîné et
> partagé, dropout, augmentation légère, arrêt anticipé sur la perte de validation, et un jeu de test jamais regardé.

**Q27. Pourquoi retourner l'image de l'œil gauche ?**
> Les deux yeux sont symétriques : le côté nasal est à gauche pour l'un, à droite pour l'autre. En retournant l'œil
> gauche, une même zone anatomique est toujours au même endroit de l'image. Un test vérifie ce miroir pixel à pixel.

**Q28. Pourquoi ne pas modifier les couleurs pour augmenter les données ?**
> Parce que sur une carte de topographie, la couleur EST la mesure (rouge = cornée bombée ou fine). Changer les
> couleurs reviendrait à changer le diagnostic.

**Q29. Vos résultats sont-ils reproductibles ?**
> Oui : graine aléatoire fixe, configuration copiée dans chaque dossier de résultats, commit Git enregistré dans le
> modèle. Un test automatique vérifie que deux entraînements identiques donnent la même perte.

**Q30. Comment être sûr d'utiliser le modèle comme à l'entraînement ?**
> La préparation des images (taille, miroir) est enregistrée DANS le fichier du modèle et relue automatiquement.
> C'est une source d'erreur classique en production que notre conception rend impossible.

---

## Qualité et sécurité

**Q31. Comment garantissez-vous la sécurité du code ?**
> À chaque commit et dans la CI : règles de sécurité **Bandit** (secrets en dur, exécution de commandes, `pickle`…),
> détection de clés privées, et **pip-audit** qui compare nos dépendances aux bases de vulnérabilités publiques.
> **Dependabot** propose chaque semaine les mises à jour de sécurité. La CI n'a que des droits de lecture et ses
> actions sont épinglées par empreinte (SHA), pour qu'on ne puisse pas les remplacer en douce.

**Q32. Un fichier modèle peut-il être dangereux ?**
> Oui : un fichier PyTorch classique peut contenir du code qui s'exécute au chargement. Nous chargeons nos modèles
> avec `weights_only=True`, qui n'accepte que des poids et des valeurs simples.

**Q33. Avez-vous des vulnérabilités connues ?**
> Oui, et nous les assumons : PyTorch 2.6 a des avis de sécurité publiés (surtout des plantages sur des opérations
> précises). L'outil tourne en local et ne charge que nos propres modèles. La mise à jour est planifiée après le
> hackathon, avec ré-entraînement. C'est documenté dans SECURITY.md. (pip-audit a aussi trouvé une version vulnérable
> de setuptools dans notre environnement : corrigée.)

**Q34. Pourquoi fixer les versions exactes des outils ?**
> Pour que le PC du développeur, pre-commit et la CI vérifient exactement la même chose. Nous l'avons appris à nos
> dépens : la CI installait une version plus récente de PyTorch dont les types diffèrent, et rejetait un code accepté
> sur notre PC. Aujourd'hui : versions identiques partout, et une commande `python scripts/check.py` qui reproduit la CI.

---

## Étape 5 — Les résultats

**Q35. Quels sont vos résultats ?**
> Sur 267 yeux jamais vus (143 patients) : **95,6 % des kératocônes détectés** (fourchette 91–99 %), **99,4 % de
> spécificité**, soit 4 kératocônes manqués sur 91 et **une seule fausse alerte sur 176** yeux sains. AUC 0,996.

**Q36. 99 %, ce n'est pas trop beau pour être vrai ?**
> C'est une bonne question. Trois raisons de rester prudents : 1) CornOrb contient surtout des cas nets (sains contre
> kératocônes avérés), pas de formes débutantes ; 2) un seul appareil, une seule population ; 3) 267 yeux, d'où nos
> fourchettes de confiance. Mais le protocole est propre : seuil choisi sur la validation, test jamais touché, découpage
> par patient. La vraie validation viendra des cas de notre ophtalmologue.

**Q37. Faites-vous mieux que les indices de l'appareil ?**
> Nous faisons au moins aussi bien que les chiffres (Kmax, épaisseur…) avec **4 fois moins de fausses alertes**. Mais les
> fourchettes se chevauchent : nous ne pouvons pas dire que c'est *statistiquement* meilleur sur 267 yeux. Et surtout,
> leurs erreurs sont **complémentaires** : 3 des 4 kératocônes que nous manquons sont attrapés par les chiffres.
> Notre prochaine étape est donc de **combiner** les deux.

**Q38. Comment avez-vous choisi le seuil ?**
> Sur la validation, jamais sur le test : le seuil le plus prudent qui garantit 95 % de sensibilité, placé au milieu
> de l'écart entre deux scores pour être robuste. Puis appliqué tel quel au test. Il est configurable : un centre
> LASIK pourrait viser 99 %.

**Q39. Qu'est-ce qu'un intervalle de confiance et pourquoi « par patient » ?**
> C'est la fourchette dans laquelle se trouve probablement la vraie performance. Nous la calculons en rejouant
> l'évaluation sur 2 000 tirages au hasard du jeu de test. Nous tirons des **patients** (avec leurs 2 yeux) car leurs
> deux yeux se ressemblent : les tirer séparément donnerait une fourchette faussement étroite (un test le démontre).

**Q40. Parlez-nous d'une erreur de votre modèle.**
> Notre seule fausse alerte, 66NY2 OS, est étiquetée « normale » mais sa carte montre une zone inférieure plus bombée
> que la supérieure, ce qui peut évoquer une forme débutante. Le modèle la signale à 99,96 %. Ce n'est peut-être pas une
> erreur mais un cas limite : nous le soumettrons à notre ophtalmologue, sans conclure nous-mêmes. C'est exactement le
> rôle d'un outil d'aide au dépistage.

**Q41. Pourquoi n'évaluez-vous le test qu'une fois ?**
> Si l'on regarde le test, qu'on modifie le modèle, puis qu'on re-teste, on finit par ajuster le modèle au test et le
> score devient optimiste. Notre outil refuse une 2e évaluation sauf avec `--force` explicite.

**Q42. Avez-vous essayé de combiner images et chiffres ?**
> Oui, par un vote moyen sans paramètre appris, seuil réglé sur la validation. La fusion classe parfaitement le test
> (AUC 1,000), mais au seuil prévu elle manque un kératocône de plus que les images seules. Nous ne l'adoptons donc pas
> comme score principal. Nous l'annonçons honnêtement : l'idée venait de l'analyse des erreurs du test, il faudra la
> confirmer sur de nouvelles données (celles de notre ophtalmologue).

**Q43. Pourquoi la fusion n'a-t-elle pas mieux marché alors que son AUC est parfaite ?**
> Parce qu'un bon classement ne suffit pas : il faut aussi savoir où placer le seuil. Sur la validation, nos deux modèles
> ne sont jamais en désaccord, donc rien n'indique où mettre la frontière pour les cas où ils se contredisent. Sur le
> test, ces désaccords tombent tous autour de 0,5, pile sur le seuil. Leçon : un désaccord entre deux experts est une
> information en soi. Dans l'outil, il déclenche le niveau « À surveiller ».

---

## Étape 6 — Expliquer et savoir dire « je ne sais pas »

**Q44. Comment le médecin sait-il POURQUOI l'outil dit « risque élevé » ?**
> Par la carte de chaleur (Grad-CAM) : les zones en orange sont celles qui ont fait monter le score. Et nous indiquons la
> part de chaque carte dans la décision, par exemple « élévation antérieure 39 %, postérieure 33 % ». Le médecin sait
> où regarder. C'est une aide à la lecture, pas une preuve : c'est écrit sur chaque figure.

**Q45. Que se passe-t-il si on donne au logiciel une image d'un autre appareil ?**
> Un réseau de neurones donne toujours un score, même absurde. Nous avons ajouté un détecteur d'images inhabituelles : il
> compare l'« empreinte » de l'œil à celles des yeux d'entraînement. En simulant un autre appareil (échelle de couleurs
> décalée), sur 43 cas où le modèle se serait trompé, 42 sont signalés « score non fiable ». Sur de vrais yeux du même
> appareil, il ne signale que 1,9 % des cas.

**Q46. Et si l'image est mauvaise (floue, noire, en noir et blanc) ?**
> Le contrôle qualité la refuse AVANT toute analyse, avec la raison : « Pachymétrie : image sans couleurs ». Aucun score
> n'est donné. Aucune vraie carte CornOrb n'a été refusée par ces règles.

**Q47. Votre carte de chaleur est-elle précise ?**
> Non, et nous le disons : elle est calculée sur une grille de 8×8, donc floue, et elle déborde parfois de la cornée.
> Elle indique la région et la carte qui ont pesé, pas un contour précis du cône.

**Q48. Comment un médecin utilise-t-il l'outil sur une consultation chargée ?**
> Il dépose les examens du jour ; l'outil rend une liste triée par urgence : d'abord les cas à revoir (images refusées
> ou inhabituelles), puis risque élevé, à surveiller, faible. Chaque œil a sa figure explicative.

---

## Étape 7 — L'application

**Q49. Comment un médecin utilise-t-il concrètement votre outil ?**
> Il ouvre une page web, glisse les 4 cartes exportées de son appareil, et obtient en quelques secondes un score, un
> niveau de risque, les zones suspectes sur chaque carte et une phrase d'explication. S'il saisit Kmax et l'épaisseur
> minimale, il a aussi l'avis des chiffres. Un deuxième onglet compare deux examens pour suivre la progression.

**Q50. Pourquoi demander Kmax et l'épaisseur alors que vous avez les images ?**
> Parce que nos analyses ont montré que les erreurs des images et des chiffres sont complémentaires. Exemple réel :
> le kératocône G3H95 OD est manqué par les images (0 %) mais les chiffres disent 84 %. L'outil détecte ce désaccord et
> classe l'œil « à surveiller » au lieu de « faible ». C'est un filet de sécurité.

**Q51. Votre application est-elle sécurisée ?**
> Les fichiers sont vérifiés en les décodant (pas sur leur extension), limités en taille, jamais utilisés comme chemin,
> ré-encodés pour effacer les métadonnées, et supprimés après chaque analyse : rien n'est conservé. La page a une
> politique de sécurité stricte (CSP) et n'affiche jamais de texte du serveur comme du code. Par défaut, le serveur
> n'écoute que sur l'ordinateur lui-même. Limite assumée : pas d'authentification, il faudrait l'ajouter avant un
> déploiement réseau.

**Q52. Pourquoi une API en plus de l'interface ?**
> Pour que d'autres logiciels (dossier patient, logiciel de l'appareil) puissent appeler Qarniya. L'interface web
> n'est qu'un client de l'API. La documentation est générée automatiquement sur /docs.

**Q53. Que se passe-t-il si on compare deux examens dont l'un est douteux ?**
> La comparaison est refusée : « au moins un des deux examens n'est pas fiable ». Mieux vaut pas de réponse qu'une
> fausse conclusion sur la progression, qui déciderait d'un traitement.
