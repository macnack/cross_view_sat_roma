# Eagle (proxima H100) runtime image for the fusion training pipeline.
# Mirrors ~/Github/sat_roma's proven Dockerfile -> slurm/sat_roma.def -> Singularity
# pattern (docs/eagle_hpc_slurm_ssh_guide.md in that repo): a CUDA *runtime* base
# (torch ships its own CUDA libs as pip wheels, so no -devel/nvcc needed), a
# world-readable venv (Eagle's pyxis container runtime runs as the submitting user,
# not root -- a venv under /root is invisible and silently falls back to the
# system python), and the libGL/libglib fix (opencv-python, not headless, wins the
# cv2 import once both are on the path; without libgl1 that import dies with
# "libGL.so.1: cannot open shared object file", killing every cv2 consumer).
#
# UNTESTED on Eagle as of authoring -- built and pushed by hand (this project has
# no CI), then converted to .sif via slurm/sat_roma.def, exactly as sat_roma does.
# torch is left UNPINNED (see slurm/README.md): environment.yml's 2.4/cu121 pin
# predates knowing Eagle's actual GPU target is H100 (sm_90); sat_roma's own
# lock resolves 2.13.0+cu130 there and notes the cu121-era wheels do not even
# cover sm_90 well. Confirm this reasoning (or override with a pin) before
# first real run -- flagged in slurm/README.md, not decided here.
FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    build-essential git wget curl ca-certificates \
    libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 libgomp1 \
    libgdal-dev gdal-bin \
 && rm -rf /var/lib/apt/lists/* \
 && python3.11 -m venv $VENV --system-site-packages=false \
 && chmod -R a+rX $VENV

COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY third_party/sat_roma_infer/ /app/third_party/sat_roma_infer/
COPY third_party/Dur360BEV/ /app/third_party/Dur360BEV/
WORKDIR /app

# torch: no pin (see header). Everything else: environment.yml's list, as pip
# packages (opencv-python, not -headless: see the libGL note above).
RUN pip install --no-cache-dir torch torchvision \
 && pip install --no-cache-dir \
      numpy'<2' scipy opencv-python rasterio pyproj shapely geopandas h5py \
      pyyaml tqdm matplotlib pytest \
 && pip install --no-cache-dir -e third_party/sat_roma_infer

ENV PYTHONPATH=/app/src:/app \
    SATROMA_INFER_DIR=/app/third_party/sat_roma_infer

CMD ["/bin/bash"]
