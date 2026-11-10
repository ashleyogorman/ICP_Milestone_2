#!/bin/bash
# Deploy the whole ICP system on k3s: MinIO + registry, OCR service, Prometheus + Grafana.
#     ./scripts/deploy_k8s.sh
set -e
cd "$(dirname "$0")/.."

if ! command -v k3s >/dev/null; then
    echo "== Installing k3s"
    curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644
fi
if ! command -v helm >/dev/null; then
    echo "== Installing Helm"
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

CHART_VERSION=93.1.1

echo "== Building OCR service image and importing it into k3s"
docker build -f service/Dockerfile -t icp-ocr-service .
docker save icp-ocr-service | sudo k3s ctr images import -

echo "== MinIO (model registry)"
kubectl apply -f k8s/minio.yaml
kubectl rollout status deployment/minio --timeout=180s

echo "== Filling the registry"
kubectl delete job upload-variants --ignore-not-found
kubectl apply -f k8s/upload-variants-job.yaml
kubectl wait --for=condition=complete job/upload-variants --timeout=600s

echo "== OCR service"
kubectl apply -f k8s/ocr.yaml
kubectl rollout status deployment/ocr --timeout=300s

echo "== Prometheus + Grafana"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
    -n monitoring --create-namespace \
    -f k8s/prometheus-config.yaml --version "$CHART_VERSION" --wait

kubectl apply -f k8s/ocr-podmonitor.yaml

echo "== Grafana dashboard"
kubectl create configmap icp-dashboard -n monitoring \
    --from-file=k8s/grafana/icp-dashboard.json --dry-run=client -o yaml | kubectl apply -f -
kubectl label configmap icp-dashboard -n monitoring grafana_dashboard=1 --overwrite

echo
echo "Done. Open Grafana with:"
echo "  kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80"
echo "  (and from your laptop: ssh -L 3000:localhost:3000 cc@<floating-ip>), then http://localhost:3000"