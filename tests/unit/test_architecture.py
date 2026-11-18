"""Test d'architecture : vérifie automatiquement que les couches respectent leurs dépendances.
Si quelqu'un importe torch ou pandas dans le domaine, ce test échoue et la CI devient rouge."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "qarniya"

# Le domaine n'a droit qu'à la bibliothèque standard et à lui-même.
ALLOWED_IN_DOMAIN = {
    "__future__", "collections", "dataclasses", "datetime", "enum", "pathlib", "types", "typing",
    "qarniya",
}  # fmt: skip


def imported_roots(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_domain_has_no_technical_dependency() -> None:
    violations = {
        f.name: sorted(imported_roots(f) - ALLOWED_IN_DOMAIN)
        for f in (PACKAGE / "domain").glob("*.py")
        if imported_roots(f) - ALLOWED_IN_DOMAIN
    }
    assert not violations, f"Le domaine doit rester pur, imports interdits : {violations}"


def test_domain_only_imports_qarniya_domain() -> None:
    for f in (PACKAGE / "domain").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("qarniya")
            ):
                assert node.module.startswith("qarniya.domain"), f"{f.name} importe {node.module}"
