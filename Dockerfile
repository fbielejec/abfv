# syntax=docker/dockerfile:1

# ---- Stage A: build the abfv Rust binary ----
# Pinned to 1.91.1 to match rust-toolchain.toml (keep the two in sync). The
# floor is 1.88, required by `ort` 2.0.0-rc.12 (ONNX inference). ort uses
# `load-dynamic`, so libonnxruntime is NOT needed at build time — it is
# dlopen'd at runtime from ABFV_ORT_DYLIB (see the runtime stage).
FROM rust:1.91.1-slim-bookworm AS rust-builder
WORKDIR /build
# Cache deps: copy manifests first (rust-toolchain.toml so the pin is honored)
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs && cargo build --release || true
# Real sources
COPY src ./src
RUN touch src/main.rs && cargo build --release

# ---- Stage A2: fetch the onnxruntime shared lib (stock Microsoft prebuilt,
#      not vendored in git) using the SAME script as local dev, so the version,
#      URL and sha256 have a single source of truth: vendored/fetch-onnxruntime.sh.
#      Resolve the symlink chain to a stable filename for the runtime COPY.
FROM debian:bookworm-slim AS ort-fetch
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY vendored/fetch-onnxruntime.sh /tmp/fetch-onnxruntime.sh
# Empty build-args -> the script's pinned defaults apply (override version AND
# sha256 together to bump): docker build --build-arg ORT_VERSION=... --build-arg ORT_SHA256=...
ARG ORT_VERSION=
ARG ORT_SHA256=
RUN ORT_VERSION="${ORT_VERSION}" ORT_SHA256="${ORT_SHA256}" ORT_DIR=/opt/ort \
      bash /tmp/fetch-onnxruntime.sh \
 && cp "$(readlink -f /opt/ort/lib/libonnxruntime.so)" /opt/libonnxruntime.so

# ---- Stage B: minimal runtime ----
# Structure prediction now runs in-process via ONNX (ort), so the runtime no
# longer carries the torch/openmm/abodybuilder3 conda stack — only a light
# Python is kept for the matplotlib `visualize` step. The vendored prebuilts:
#   * freesasa     2.1.3, built --disable-json --disable-xml (libstdc++/libm/libc only)
#   * abb3.onnx    ABodyBuilder3 exported to ONNX (workers/export_onnx.py)
#   * libonnxruntime.so  Microsoft's 1.24 manylinux build (glibc >= 2.28, ok on bookworm)
# (workers/predict.py is copied for reference but is NOT executed here — its
# torch/abb3 imports are intentionally absent from this image.)
FROM python:3.12-slim-bookworm AS runtime

# Runtime shared libs: libgomp1 for onnxruntime's OpenMP, libstdc++6 for
# freesasa + onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# Only the visualize step needs Python deps.
RUN pip install --no-cache-dir matplotlib pandas

# abfv binary from rust-builder; libonnxruntime from the fetch stage; the model
# + freesasa are vendored prebuilts copied from the build context.
COPY --from=rust-builder /build/target/release/abfv /opt/abfv/bin/abfv
COPY --from=ort-fetch /opt/libonnxruntime.so /opt/abfv/lib/libonnxruntime.so
COPY vendored/freesasa  /opt/abfv/bin/freesasa
COPY vendored/abb3.onnx /opt/abfv/model/abb3.onnx
COPY workers/  /opt/abfv/workers/
COPY examples/ /opt/abfv/examples/

# Container paths for the abfv CLI (clap reads these ABFV_* env vars).
ENV ABFV_ONNX=/opt/abfv/model/abb3.onnx \
    ABFV_ORT_DYLIB=/opt/abfv/lib/libonnxruntime.so \
    ABFV_PYTHON=/usr/local/bin/python \
    ABFV_VISUALIZE=/opt/abfv/workers/visualize.py \
    ABFV_FREESASA=/opt/abfv/bin/freesasa

# Fail the build if any baked artifact is missing, freesasa links in this clean
# base, the visualize deps import, and the whole pipeline runs end-to-end
# (proves load-dynamic resolves libonnxruntime, the ONNX model runs, and the
# Rust -> freesasa -> contacts -> matplotlib chain works).
RUN test -x /opt/abfv/bin/abfv \
 && test -x /opt/abfv/bin/freesasa \
 && test -f /opt/abfv/model/abb3.onnx \
 && test -f /opt/abfv/lib/libonnxruntime.so \
 && /opt/abfv/bin/freesasa --version \
 && python -c "import matplotlib, pandas; print('visualize deps OK')" \
 && mkdir -p /tmp/smoke \
 && /opt/abfv/bin/abfv --out-dir /tmp/smoke \
      --heavy-file /opt/abfv/examples/heavy.fasta \
      --light-file /opt/abfv/examples/light.fasta \
 && test -s /tmp/smoke/contacts.csv \
 && rm -rf /tmp/smoke \
 && echo "pipeline smoke test OK"

WORKDIR /work
ENTRYPOINT ["/opt/abfv/bin/abfv"]
