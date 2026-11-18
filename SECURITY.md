# Sécurité et données

## Données de santé (loi 09-08)
- **Aucune donnée patient dans Git** : `/data/` et `/outputs/` sont exclus (`.gitignore`), et le hook
  `check-added-large-files` bloque l'ajout accidentel d'images ou de modèles.
- Les patients sont identifiés par un **code anonyme** ; jamais de nom, numéro de dossier ou date de naissance.
- Les données du médecin restent sur le poste de l'équipe, utilisées uniquement pour le projet, supprimées sur demande.

## Mesures dans le code
| Risque | Mesure |
|---|---|
| Fichier modèle piégé (code caché dans un `.pt`) | `torch.load(..., weights_only=True)` : seuls des poids et des valeurs simples sont lus |
| Commande système détournée | `git` appelé par son chemin complet (`shutil.which`), commande fixe, délai maximal |
| Configuration malveillante ou erronée | YAML lu avec `yaml.safe_load` + validation stricte pydantic (clé inconnue = erreur) |
| Secrets dans le code | hook `detect-private-key` + règles de sécurité Bandit (ruff `S`) à chaque commit |
| Vulnérabilités des dépendances | `pip-audit` en CI + Dependabot chaque semaine |
| Fuite de données par le conteneur | utilisateur non-root, système de fichiers en lecture seule, `/tmp` en mémoire, aucun privilège, port sur 127.0.0.1 |
| Identité dans le rapport PDF | référence anonyme seulement (lettres, chiffres, `-`, `_`), noms et dates refusés |
| CI compromise | permissions en lecture seule, actions épinglées par SHA, identifiants Git non conservés |

## Points connus (18/11/2026)
- **PyTorch 2.6.0** : 12 avis de sécurité publiés, la plupart corrigés dans des versions plus récentes (jusqu'à 2.13),
  certains sans correctif. Il s'agit surtout de plantages (déni de service) sur des opérations précises ou du
  chargement de modèles non fiables. **Atténuation** : usage local, nous ne chargeons que nos propres modèles, en
  `weights_only=True`. **Action prévue** : mise à jour de PyTorch après le hackathon, avec ré-entraînement et tests.
- PyTorch installé depuis l'index officiel PyTorch (versions `+cu124` / `+cpu`) n'est pas sur PyPI :
  `pip-audit` ne peut pas l'auditer automatiquement. Contrôle manuel :
  `pip-audit --no-deps --disable-pip -r <fichier contenant torch==2.6.0>`.

## Signaler un problème
Merci de ne pas ouvrir d'issue publique pour une faille : contactez l'équipe NeuroLens directement.
