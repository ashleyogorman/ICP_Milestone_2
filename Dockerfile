FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Added tesseract-ocr-chi-sim for Chinese language support in Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential software-properties-common curl wget git unzip nano vim \
    python3 python3-pip python3-venv python3-dev \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /icp-milestone-2

COPY README.md .
COPY requirements-icp.txt .

# Install Jupyter and dependencies from requirements file
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-icp.txt

RUN python3 -c "import easyocr; easyocr.Reader(['ch_sim'], gpu=False)"

RUN pip install --no-cache-dir jupyter

COPY milestone_2.ipynb .

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
