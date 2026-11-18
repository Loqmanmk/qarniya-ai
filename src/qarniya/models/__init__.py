"""Architectures de réseaux, registre de modèles, préparation des images et sauvegarde."""

from qarniya.models.checkpoint import LoadedModel, load_checkpoint, save_checkpoint
from qarniya.models.multimap import BACKBONES, MultiMapClassifier
from qarniya.models.novelty import NoveltyDetector
from qarniya.models.predictor import Predictor
from qarniya.models.preprocessing import PreprocessingSpec, Preprocessor
from qarniya.models.registry import ModelSpec, available_models, build_model, register_model

__all__ = [
    "BACKBONES",
    "LoadedModel",
    "ModelSpec",
    "MultiMapClassifier",
    "NoveltyDetector",
    "Predictor",
    "PreprocessingSpec",
    "Preprocessor",
    "available_models",
    "build_model",
    "load_checkpoint",
    "register_model",
    "save_checkpoint",
]
