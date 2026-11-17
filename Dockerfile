# GFPGAN inference container (CPU). Build from the repository root:
#   docker build -t gfpgan .
#   docker run --rm -v "$PWD/inputs:/app/inputs" -v "$PWD/results:/app/results" gfpgan \
#       -i inputs/whole_imgs -o results -v 1.4 -s 2
# Models are downloaded automatically on first use; mount a volume on
# gfpgan/weights and experiments/pretrained_models to cache them.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# basicsr still declares tb-nightly, which is unavailable on some Python
# versions: install its real runtime deps manually, then install
# basicsr/facexlib/realesrgan without resolving deps.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir addict future lmdb opencv-python-headless pyyaml scipy tensorboard tqdm yapf requests scikit-image imageio tifffile lazy-loader packaging \
    && pip install --no-cache-dir basicsr==1.4.2 facexlib==0.3.0 realesrgan==0.3.0 --no-deps

COPY gfpgan/ gfpgan/
COPY inference_gfpgan.py ./
RUN pip install --no-cache-dir --no-deps .

ENTRYPOINT ["python", "inference_gfpgan.py"]
