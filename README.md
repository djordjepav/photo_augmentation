# Photo Augmentation

This is a demo project for augmentation of photos using Stable Difussion models. 
TODO: Add more details, specific tasks being solved, etc.

## Prerequisites

- Docker (for containerized deployment)  
- NVIDIA GPU with CUDA 11.7+ (recommended)  
- Python 3.8+ (for native installation)  

## Installation

### Option 1: Docker Deployment (Recommended)

Build the Docker image
```bash
docker build -t people-augmentation .
```

Run the container with GPU support
```bash
docker run --gpus all -p 8000:8000 people-augmentation
```

The API will be available at http://localhost:8000.

### Option 2: Native Python Installation

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac)
venv\Scripts\activate     # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

