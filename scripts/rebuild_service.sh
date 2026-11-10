#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker build -f service/Dockerfile -t icp-ocr-service .
docker save icp-ocr-service | sudo k3s ctr images import -
kubectl rollout restart deployment/ocr
kubectl rollout status deployment/ocr