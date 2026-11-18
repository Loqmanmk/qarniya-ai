"""Outils partagés par les adaptateurs pour construire et reconnaître les chemins d'images."""

from __future__ import annotations

import re
from pathlib import Path

from qarniya.data.config import DatasetConfig
from qarniya.domain import MapType


def build_map_paths(config: DatasetConfig, patient: str, eye_raw: str) -> dict[MapType, Path]:
    """Chemins attendus des 4 cartes d'un œil, d'après `image_pattern`."""
    return {
        map_type: config.root / config.image_pattern.format(patient=patient, eye=eye_raw, map=token)
        for map_type, token in config.map_tokens.items()
    }


def pattern_to_regex(config: DatasetConfig) -> re.Pattern[str]:
    """Transforme `{patient}_{eye}_{map}.png` en expression régulière qui extrait les 3 champs.

    Les valeurs possibles de {eye} et {map} sont connues (YAML) : on les liste explicitement,
    ce qui rend le découpage sans ambiguïté même si le code patient contient des « _ ».
    """

    def alternatives(words: list[str]) -> str:
        return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))

    regex = re.escape(config.image_pattern)
    regex = regex.replace(re.escape("{patient}"), r"(?P<patient>[^/]+?)")
    regex = regex.replace(re.escape("{eye}"), f"(?P<eye>{alternatives(list(config.eye_values))})")
    regex = regex.replace(
        re.escape("{map}"), f"(?P<map>{alternatives(list(config.map_tokens.values()))})"
    )
    return re.compile(f"^{regex}$", flags=re.IGNORECASE)
