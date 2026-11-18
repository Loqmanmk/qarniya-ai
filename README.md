# Qarniya AI

**Aide au dépistage précoce du kératocône à partir des cartes de topographie cornéenne.**
Équipe NeuroLens · Hackathon Nexus HealthTech 2026 · EHEIM Oujda

> ⚠️ Outil d'aide à la décision. Il ne pose pas de diagnostic : la décision reste celle de l'ophtalmologue.

## Ce que fait Qarniya AI
À partir des 4 cartes d'un œil (axiale, élévation antérieure, élévation postérieure, pachymétrie) :
1. un **score de risque** de kératocône et un niveau (faible / à surveiller / élevé) ;
2. une **carte de chaleur** qui montre la zone suspecte ;
3. la **progression** entre deux examens du même œil ;
4. un **refus explicite** quand l'entrée n'est pas exploitable.

## Démarrage rapide (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev,serve]"
pre-commit install
qarniya version
```

## Avant chaque push : une seule commande
```powershell
python scripts/check.py
```
Elle lance **exactement** les vérifications de la CI (lint + sécurité, formatage, types, tests, dépendances vulnérables).
Tout vert = vous pouvez pousser. Voir aussi [SECURITY.md](SECURITY.md).

## Commandes de qualité (détail)
| Commande | Rôle |
|---|---|
| `ruff check .` | détecte erreurs et mauvaises pratiques |
| `ruff format .` | formate le code automatiquement |
| `mypy` | vérifie les types |
| `pytest` | lance les tests |
| `pytest --cov` | tests + couverture de code |
| `pre-commit run --all-files` | tout vérifier comme avant un commit |
| `pip-audit --skip-editable` | cherche les vulnérabilités connues des dépendances |

## Organisation du dépôt
```
configs/        configuration YAML (datasets, modèles, entraînement)
src/qarniya/    le paquet Python (voir docs/02_architecture.md)
tests/          tests unitaires et d'intégration
app/            interface web
scripts/        scripts utilitaires
docs/           cadrage, architecture, décisions, préparation du jury
prototype/      premier prototype du 18/11 (référence, non maintenu)
```

## Documentation
- [Cadrage du projet](docs/01_cadrage.md)
- [Architecture](docs/02_architecture.md)
- [Résultats sur le jeu de test](docs/05_evaluation.md)
- [Questions du jury](docs/jury_questions.md)
- [Sécurité et données](SECURITY.md)

## Données
Dataset CornOrb — Lazouni M.E.A. et al., Zenodo 2025, DOI [10.5281/zenodo.17121301](https://doi.org/10.5281/zenodo.17121301), licence CC BY 4.0.
Les données ne sont jamais versionnées dans Git (voir `.gitignore`).
