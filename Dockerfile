FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
RUN wget https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt
RUN wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install git+https://github.com/tencent-ailab/IP-Adapter.git

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
