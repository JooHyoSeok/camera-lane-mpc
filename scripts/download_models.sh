#!/usr/bin/env bash
# Download lane-detection weights into models/.
#   culane_18.onnx     UFLD ResNet18 trained on CULane (official weights, ONNX export by ailia-models)
#   culane_18_md.onnx  the same model fine-tuned on auto-labeled MetaDrive frames (this repo's release)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

REPO="${REPO:-JooHyoSeok/camera-lane-mpc}"
RELEASE="${RELEASE:-v0.1.0}"

fetch() {  # url dest
  if [ -f "$2" ]; then echo "exists: $2"; return; fi
  echo "downloading $2"
  curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}

fetch "https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/culane_18.onnx" models/culane_18.onnx
fetch "https://github.com/${REPO}/releases/download/${RELEASE}/culane_18_md.onnx" models/culane_18_md.onnx
