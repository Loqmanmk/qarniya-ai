"""Valeurs fixes du domaine. Utiliser des Enum plutôt que des chaînes évite les fautes de frappe
("OD" vs "od" vs "droit") et rend les erreurs visibles dès l'écriture du code."""

from enum import Enum


class EyeSide(str, Enum):
    """Œil examiné (abréviations ophtalmologiques latines)."""

    RIGHT = "OD"  # oculus dexter
    LEFT = "OS"  # oculus sinister


class MapType(str, Enum):
    """Les 4 cartes de topographie cornéenne d'un examen."""

    AXIAL = "axial"  # courbure de la face avant
    ANTERIOR_ELEVATION = "anterior"  # élévation de la face avant
    POSTERIOR_ELEVATION = "posterior"  # élévation de la face arrière (souvent la 1re touchée)
    PACHYMETRY = "pachymetry"  # épaisseur de la cornée


class Diagnosis(str, Enum):
    """Étiquette posée par le médecin (vérité terrain)."""

    NORMAL = "normal"
    SUSPECT = "suspect"
    KERATOCONUS = "keratoconus"


class RiskLevel(str, Enum):
    """Niveau de risque affiché au médecin."""

    LOW = "faible"
    MONITOR = "a_surveiller"
    HIGH = "eleve"
