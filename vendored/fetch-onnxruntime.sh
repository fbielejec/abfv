#!/usr/bin/env bash
# Fetch the onnxruntime shared library that `abfv` dlopens at runtime
# (ort `load-dynamic`, ABFV_ORT_DYLIB). The lib is a stock Microsoft
# redistributable, so it is NOT committed — this script populates
# vendored/onnxruntime/ on demand (and the Dockerfile fetches it the same way).
#
# Usage:
#   vendored/fetch-onnxruntime.sh            # default version -> vendored/onnxruntime
#   ORT_VERSION=1.24.2 ORT_DIR=/some/dir vendored/fetch-onnxruntime.sh
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.24.2}"
ORT_DIR="${ORT_DIR:-$(cd "$(dirname "$0")" && pwd)/onnxruntime}"
# sha256 of lib/libonnxruntime.so.${ORT_VERSION} (linux x64). Update on bump.
ORT_SHA256="${ORT_SHA256:-ffc84d48e845cf0b562ba4ea5ca32aaafc0d4069019fef4f63095b307d0270ad}"

lib="${ORT_DIR}/lib/libonnxruntime.so.${ORT_VERSION}"
if [ -f "$lib" ] && echo "${ORT_SHA256}  ${lib}" | sha256sum -c - >/dev/null 2>&1; then
    echo "onnxruntime ${ORT_VERSION} already present at ${ORT_DIR}"
    exit 0
fi

url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading ${url}"
curl -fsSL "$url" -o "$tmp/ort.tgz"
mkdir -p "$ORT_DIR"
tar xzf "$tmp/ort.tgz" -C "$ORT_DIR" --strip-components=1

echo "${ORT_SHA256}  ${lib}" | sha256sum -c -
echo "onnxruntime ${ORT_VERSION} -> ${ORT_DIR}/lib"
