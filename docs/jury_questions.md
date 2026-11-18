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
