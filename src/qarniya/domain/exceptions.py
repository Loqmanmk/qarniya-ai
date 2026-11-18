"""Erreurs métier. Une hiérarchie propre permet à l'API de les traduire en messages clairs
pour le médecin, au lieu d'une trace Python incompréhensible."""


class QarniyaError(Exception):
    """Classe mère de toutes les erreurs du projet."""


class InvalidExamError(QarniyaError):
    """L'examen est incohérent ou inexploitable."""


class MissingMapError(InvalidExamError):
    """Une ou plusieurs des 4 cartes obligatoires manquent."""
