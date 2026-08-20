# StressViz — runs as-is on Google Cloud Run, on Hugging Face Spaces (Docker
# SDK), and on any normal container host.
# See DEPLOY_CLOUDRUN.md or DEPLOY_SPACE.md.
FROM python:3.11-slim

# OpenCV / Triangle runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential \
    libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000, not root, and anything owned by root
# is then read-only to the process. Creating that user here and giving it /app
# is the difference between a working Space and a permission error on the first
# write. HOME and the cache vars matter for the same reason: gmsh, matplotlib
# and pip all write to $HOME by default, and root's home is not writable here.
RUN useradd -m -u 1000 sv
ENV HOME=/home/sv \
    PATH=/home/sv/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app
RUN mkdir -p /app/outputs && chown -R sv:sv /app
ENV OUTPUT_DIR=/app/outputs

COPY --chown=sv:sv requirements.txt .
USER sv
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=sv:sv app ./app
COPY --chown=sv:sv static ./static
# The knowledge base and the pocketing calibration. Without data/ the deployed
# assistant starts with an empty library and cites nothing, which reads as a
# broken bot rather than an unpopulated one. data/kb.json is git-ignored, so on
# a Space it has to be force-added once -- DEPLOY_SPACE.md says how.
COPY --chown=sv:sv data ./data
# Not needed to serve, but tools/ is what rebuilds the library, and shipping it
# means the deployed Space can re-ingest without a local checkout.
COPY --chown=sv:sv tools ./tools

# Written at run time by app/save.py. A Space's disk is not persistent, so this
# lasts until the next restart and no longer -- mount a volume if the saved
# images have to outlive a rebuild.
EXPOSE 8000

# Cloud Run does not let the container choose its port -- it sets $PORT (8080)
# and expects to be listened to there. Spaces and a local run set nothing, hence
# the 8000 default, so one image serves all three. The shell form is what makes
# ${PORT:-8000} expand at all; `exec` is what keeps uvicorn as PID 1 so a stop
# signal reaches it instead of the shell, which is the difference between a
# clean shutdown and a ten-second wait for the kill.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
