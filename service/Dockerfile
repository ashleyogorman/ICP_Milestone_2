FROM ubuntu:22.04
 
ENV DEBIAN_FRONTEND=noninteractive
 
# Python, OpenCV's system library, and Tesseract with Chinese (simplified)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /icp-milestone-2
 
COPY requirements-icp.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-icp.txt
 
# Only the code the service needs
COPY icp/ icp/
COPY service/ service/
COPY registry/ registry/
 
# So "import icp" works from service/server.py
ENV PYTHONPATH=/icp-milestone-2
 
# gRPC and Prometheus /metrics
EXPOSE 50051 8000
 
CMD ["python3", "service/server.py"]