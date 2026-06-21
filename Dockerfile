# Schematic AI Reasoner — app image (Streamlit UI + extraction pipeline + detector).
#
# Heavy deps (torch + ultralytics for the YOLO detector) are installed via the
# [detector] extra. If you want a lighter image without the detector, drop the
# `--extra detector` line below; the app falls back to the geometric pipeline
# automatically when torch is absent.
FROM python:3.12-slim AS base

# pymupdf needs libGL at runtime; OpenCV (detector extra) needs libglib too.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching than copying source).
COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir -e ".[dev,detector]"

# Copy the source tree.
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# YOLO weights are NOT in the repo (too big). Mount them at runtime as a volume
# at /app/runs/detect/... or pass --weights to the detector runner. If absent,
# the pipeline silently falls back to the geometric path.
RUN mkdir -p /app/runs/detect /app/.cache

EXPOSE 8501

# Healthcheck: Streamlit responds on / when ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
