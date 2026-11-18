# Image de Qarniya AI (API + interface web), PyTorch CPU : tourne sur n'importe quel PC ou serveur.
#
#   docker build -t qarniya-ai .
#   docker run --rm -p 127.0.0.1:8765:8000 -v "<dossier du modèle>:/model:ro" qarniya-ai
#   → http://127.0.0.1:8765
#
# Le modèle n'est PAS dans l'image : il est monté en lecture seule (données de santé hors image,
# et on peut changer de modèle sans reconstruire).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# 1) PyTorch CPU en premier : couche lourde, mise en cache tant que la version ne change pas
RUN pip install --upgrade pip "setuptools>=84.1" \
 && pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# 2) le code du projet (le .dockerignore n'envoie que pyproject, README et src/)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# 3) utilisateur sans privilèges : l'application ne tourne jamais en root
RUN useradd --create-home --uid 10001 qarniya
USER qarniya

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/api/health').status != 200)"

# 0.0.0.0 À L'INTÉRIEUR du conteneur ; c'est le « -p 127.0.0.1:... » de docker run qui limite l'accès au PC
ENTRYPOINT ["qarniya", "serve", "/model", "--host", "0.0.0.0", "--port", "8000"]
