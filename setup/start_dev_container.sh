#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker build -t icp-pipeline .
docker rm -f icp 2>/dev/null || true
docker run -d --name icp --shm-size=2g -p 8888:8888 -v "$(pwd)":/icp-milestone-2 icp-pipeline
sleep 5
docker logs icp 2>&1 | grep token || echo "Jupyter starting; run: docker logs icp 2>&1 | grep token"