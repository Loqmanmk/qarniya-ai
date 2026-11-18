"""Rapport PDF : la référence doit rester anonyme."""

import pytest

from qarniya.domain import QarniyaError
from qarniya.services.report import validate_reference


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("  ", None), (" P017 ", "P017"),
                                               ("KC-2026_03", "KC-2026_03")])  # fmt: skip
def test_valid_anonymous_references(raw: str | None, expected: str | None) -> None:
    assert validate_reference(raw) == expected


@pytest.mark.parametrize("raw", ["Fatima Zahra", "née le 01/02/2003", "é", "a" * 25])
def test_names_dates_and_long_text_are_refused(raw: str) -> None:
    with pytest.raises(QarniyaError, match="anonyme"):
        validate_reference(raw)
