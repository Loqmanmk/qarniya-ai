"""Lance EXACTEMENT les vérifications de la CI GitHub, sur votre PC, avant de pousser.

    python scripts/check.py           # tout
    python scripts/check.py --fast    # sans les tests lents (intégration)

Code de sortie 0 = tout est vert, vous pouvez pousser.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"
ROOT = Path(__file__).resolve().parents[1]


def project_python() -> Path | None:
    """Python de l'environnement du projet (.venv), sous Windows comme sous Linux/Mac."""
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def ensure_project_python() -> None:
    """Si le script est lancé avec un autre Python (environnement non activé), on se relance
    automatiquement avec celui du projet, où sont installés ruff, mypy, pytest…"""
    venv_python = project_python()
    if venv_python is None or Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        return
    print(f"(environnement du projet non activé : relance avec {venv_python})", flush=True)
    completed = subprocess.run([str(venv_python), *sys.argv], check=False)  # noqa: S603
    sys.exit(completed.returncode)


def checks(fast: bool) -> list[tuple[str, list[str]]]:
    py = [sys.executable, "-m"]
    pytest_args = ["pytest", "--cov", "-q"] + (["-m", "not integration"] if fast else [])
    return [
        ("Lint + sécurité (ruff)", [*py, "ruff", "check", "."]),
        ("Formatage (ruff format)", [*py, "ruff", "format", "--check", "."]),
        ("Types (mypy)", [*py, "mypy"]),
        ("Tests + couverture (pytest)", [*py, *pytest_args]),
        ("Dépendances vulnérables (pip-audit)", [*py, "pip_audit", "--skip-editable"]),
    ]


def main() -> int:
    ensure_project_python()
    os.chdir(ROOT)  # fonctionne quel que soit le dossier d'où on lance le script
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="ignore les tests d'intégration")
    args = parser.parse_args()

    results: list[tuple[str, bool, float]] = []
    for name, command in checks(args.fast):
        print(f"\n{BOLD}▶ {name}{RESET}", flush=True)
        start = time.perf_counter()
        ok = subprocess.run(command, check=False).returncode == 0  # noqa: S603  (commandes fixes)
        results.append((name, ok, time.perf_counter() - start))

    print(f"\n{BOLD}Résumé{RESET}")
    for name, ok, seconds in results:
        mark = f"{GREEN}✔{RESET}" if ok else f"{RED}✘{RESET}"
        print(f"  {mark} {name:40s} {seconds:5.1f} s")
    all_ok = all(ok for _, ok, _ in results)
    verdict = (
        f"{GREEN}Tout est vert : vous pouvez pousser.{RESET}"
        if all_ok
        else (f"{RED}À corriger avant de pousser.{RESET}")
    )
    print(f"\n{verdict}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
