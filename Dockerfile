# syntax=docker/dockerfile:1

# ---- Stage A: build the abfv Rust binary ----
# Pinned to 1.85.1 to match rust-toolchain.toml (keep the two in sync). 1.85 is
# the floor because Cargo.lock resolves clap 4.6.1, whose own manifest declares
# `edition = "2024"`, which Cargo < 1.85 cannot parse.
FROM rust:1.85.1-slim-bookworm AS rust-builder
WORKDIR /build
# Cache deps: copy manifests first (rust-toolchain.toml so the pin is honored)
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release || true
# Real sources
COPY src ./src
RUN touch src/main.rs && cargo build --release

# FreeSASA is a prebuilt 2.1.3 binary that lives in `vendored/`
# (compiled with `--disable-json  --disable-xml`,
# so it is ABI-compatible with the runtime stage and needs only
# libstdc++/libm/libc). It is COPY'd straight into the runtime stage below.

# ---- Stage B: build + slim the conda env (heavy; discarded, only the env dir
#      and the checkpoint are copied forward) ----
FROM mambaorg/micromamba:1.5-bookworm AS env-builder
ARG ABB3_COMMIT=18e4058015a39c5405c08a0d5629cf302627b253
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create the conda env (cached until the yml changes)
COPY docker/environment.docker.yml /tmp/environment.docker.yml
RUN micromamba create -y -f /tmp/environment.docker.yml && micromamba clean -a -y
ENV PATH=/opt/conda/envs/abfv/bin:$PATH

# ABodyBuilder3 + minimal runtime deps (NOT abb3's heavy install_requires).
# A pip constraints file pins the conda-provided scientific stack so that no
# transitive dependency may upgrade/uninstall it. Without this:
#   - "lightning>=2.0.4" resolves to lightning 2.6 -> drags torch 2.8 + the full
#     nvidia CUDA wheel set, uninstalling the conda cpuonly torch 2.0.1, and
#   - pandas/matplotlib pull numpy 2.0.2, uninstalling conda numpy 1.21.6 and
#     breaking scipy/torchmetrics (compiled against numpy 1.x).
# lightning/torchmetrics versions are ABB3's own upstream pins
# (pinned-versions.txt at the build commit), validated against torch 2.0/2.1.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
RUN printf 'numpy==1.21.6\ntorch==2.0.1\nscipy==1.11.2\n' > /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
      "lightning==2.1.2" "lightning-utilities==0.10.0" "torchmetrics==1.2.1" \
      biopython einops dm-tree ml_collections loguru python-box tqdm \
 && pip install --no-cache-dir --no-deps \
      "git+https://github.com/Exscientia/abodybuilder3@${ABB3_COMMIT}" \
 && pip install --no-cache-dir -c /tmp/constraints.txt "pandas==1.5.3" matplotlib

# Slim the env (~4.0 GB -> ~1.7 GB). The CUDA libs are pulled in by conda-forge's
# openmm (its GPU platform plugin) but are dead weight here: torch is the CPU
# build and never links them, and we only ever IMPORT openmm (the refinement /
# simulation path is never executed). Headers + static libs are build-only.
# This runs in a separate layer from `micromamba create`, which is fine because
# the final stage COPY --from picks up only the post-slim directory state.
RUN rm -f  /opt/conda/envs/abfv/lib/libcu*.so* \
           /opt/conda/envs/abfv/lib/libnpp*.so* \
           /opt/conda/envs/abfv/lib/libnv*.so* \
 && rm -rf /opt/conda/envs/abfv/include \
 && find /opt/conda/envs/abfv -name '__pycache__' -type d -prune -exec rm -rf {} + \
 && find /opt/conda/envs/abfv -name '*.a' -delete

# Import smoke test on the SLIMMED env (TDD driver for the dep set; also proves
# the CUDA-lib removal did not break openmm/torch imports)
RUN python -c "import torch, openmm, pdbfixer; \
import abodybuilder3; \
from abodybuilder3.utils import string_to_input, output_to_pdb, add_atom37_to_output; \
from abodybuilder3.lightning_module import LitABB3; \
import pandas, matplotlib; \
print('imports OK')"

# The model checkpoint lives in `vendored/best_second_stage.ckpt`
# and is COPY'd straight into the runtime stage below.

# ---- Stage C: minimal runtime ----
# Fresh micromamba base (no apt/pip build layers); we COPY in just the slimmed
# conda env, the checkpoint, the two binaries, and the workers/examples.
FROM mambaorg/micromamba:1.5-bookworm AS runtime
USER root

# Slimmed conda env (final directory state from env-builder)
COPY --from=env-builder /opt/conda/envs/abfv /opt/conda/envs/abfv

# abfv binary from the rust-builder stage; freesasa + checkpoint from the
# vendored prebuilts; plus workers + examples
COPY --from=rust-builder /build/target/release/abfv /opt/abfv/bin/abfv
COPY vendored/freesasa                /opt/abfv/bin/freesasa
COPY vendored/best_second_stage.ckpt  /opt/abfv/model/best_second_stage.ckpt
COPY workers/  /opt/abfv/workers/
COPY examples/ /opt/abfv/examples/

ENV PATH=/opt/conda/envs/abfv/bin:$PATH
# Container paths for the abfv CLI (clap reads these ABFV_* env vars)
ENV ABFV_PYTHON=/opt/conda/envs/abfv/bin/python \
    ABFV_SCRIPT=/opt/abfv/workers/predict.py \
    ABFV_VISUALIZE=/opt/abfv/workers/visualize.py \
    ABFV_CHECKPOINT=/opt/abfv/model/best_second_stage.ckpt \
    ABFV_FREESASA=/opt/abfv/bin/freesasa

# Fail the build if any baked artifact is missing, the copied env still imports,
# and the freesasa binary runs in this clean base (links resolve).
RUN test -x /opt/abfv/bin/abfv \
 && test -x /opt/abfv/bin/freesasa \
 && test -f /opt/abfv/model/best_second_stage.ckpt \
 && test -f /opt/abfv/workers/predict.py \
 && test -f /opt/abfv/workers/visualize.py \
 && /opt/abfv/bin/freesasa --version \
 && python -c "import torch, openmm, pdbfixer; from abodybuilder3.lightning_module import LitABB3; print('runtime imports OK')"

WORKDIR /work
ENTRYPOINT ["/opt/abfv/bin/abfv"]
