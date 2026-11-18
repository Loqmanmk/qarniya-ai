"""API REST (FastAPI) et interface web, qui exposent les services d'analyse."""

from qarniya.api.app import create_app

__all__ = ["create_app"]
