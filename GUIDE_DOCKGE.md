# 🐳 HunyuanWorld-Voyager for TrueNAS Dockge

## 📋 Quick Guide for Dockge

### ⚡ MANDATORY Prerequisites

1. **NVIDIA GPU with 60GB+ VRAM** (RTX A6000, A100, H100)
2. **NVIDIA Container Toolkit installed on TrueNAS**
3. **500GB+ free space** for models

### 🚀 Installation on Dockge (Step by Step)

#### 1. Verify NVIDIA Container Toolkit

First, connect via SSH to TrueNAS and check:

```bash
# Check if nvidia-smi works
nvidia-smi

# Test Docker with GPU
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi
```

If it errors, install NVIDIA Container Toolkit:

```bash
# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(DPKG_ARCH) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### 2. Create Stack in Dockge

1. **Open Dockge** in browser: `http://[TRUENAS_IP]:5001`
2. **Click "+" to create new stack**
3. **Stack name**: `hunyuanworld-voyager`
4. **Paste the docker-compose.yml below** in the editor

#### 3. Docker Compose for Dockge

```yaml
version: '3.8'

services:
  hunyuanworld-voyager:
    image: nvidia/cuda:12.4-devel-ubuntu22.04
    container_name: hunyuanworld-voyager
    restart: unless-stopped
    
    entrypoint: ["/bin/bash", "-c"]
    command:
      - |
        set -e
        echo "🚀 HunyuanWorld-Voyager - Starting..."
        
        export DEBIAN_FRONTEND=noninteractive
        
        # Install system dependencies
        apt-get update && apt-get install -y \
            python3.11 python3.11-dev python3-pip git curl ffmpeg \
            libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
        
        # Configure Python
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        python3 -m pip install --upgrade pip
        
        # Clone repository if needed
        if [ ! -d "/app/.git" ]; then
            echo "📦 Cloning repository..."
            git clone https://github.com/dmax101/HunyuanWorld-Voyager.git /app
        fi
        
        cd /app
        
        # Install PyTorch CUDA
        echo "🔧 Installing PyTorch..."
        pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
            --index-url https://download.pytorch.org/whl/cu124
        
        # Install dependencies
        pip install -r requirements.txt
        pip install transformers==4.39.3 flash-attn xfuser==0.4.2 \
                   nvidia-cublas-cu12==12.4.5.8 scipy==1.11.4
        
        # Install additional dependencies
        pip install --no-deps git+https://github.com/microsoft/MoGe.git || true
        pip install git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38 || true
        
        # Create directories
        mkdir -p /app/{ckpts,examples,results,temp,temp_images}
        
        # Check GPU
        echo "🎮 GPU Status:"
        nvidia-smi || echo "⚠️ GPU not detected"
        
        # Check models
        if [ ! -f "/app/ckpts/config.json" ]; then
            echo ""
            echo "❌ MODELS NOT FOUND!"
            echo "📥 To download models (~100GB):"
            echo "   docker exec -it hunyuanworld-voyager bash"
            echo "   cd /app/ckpts"
            echo "   pip install huggingface_hub[cli]"
            echo "   huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./"
            echo ""
        fi
        
        echo "✅ Configuration completed!"
        echo "🌐 Interface: http://localhost:3500"
        
        # Start application
        python3 app.py
    
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    ports:
      - "3500:8080"
    
    volumes:
      - voyager_app:/app
      - voyager_models:/app/ckpts
      - voyager_results:/app/results
      - voyager_temp:/app/temp
      - voyager_cache:/root/.cache
    
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=all
      - PYTHONUNBUFFERED=1
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=8080
      - ALLOW_RESIZE_FOR_SP=1
    
    shm_size: 8gb
    
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080 || exit 1"]
      interval: 60s
      timeout: 15s
      retries: 3
      start_period: 600s

volumes:
  voyager_app:
  voyager_models:
  voyager_results:
  voyager_temp:
  voyager_cache:
```

#### 4. Deploy the Stack

1. **Click "Deploy"** in Dockge
2. **Wait for build** (first time: 20-30 minutes)
3. **Monitor logs** to see progress

#### 5. Download Models (MANDATORY)

After the container is running:

```bash
# Access the container
docker exec -it hunyuanworld-voyager bash

# Go to models directory
cd /app/ckpts

# Install HuggingFace CLI
pip install huggingface_hub[cli]

# Download models (~100GB)
huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./

# Exit container
exit
```

#### 6. Access the Interface

Open in browser: `http://[TRUENAS_IP]:3500`

## 🎯 How to Use

1. **Upload image** in the interface
2. **Choose camera direction** (forward, backward, left, right)
3. **Enter descriptive prompt**
4. **Generate video** (may take several minutes)

## 🔧 Useful Commands

```bash
# View container logs
docker logs -f hunyuanworld-voyager

# Access container shell
docker exec -it hunyuanworld-voyager bash

# Check GPU inside container
docker exec -it hunyuanworld-voyager nvidia-smi

# Restart container
docker restart hunyuanworld-voyager

# View volume status
docker volume ls | grep voyager
```

## ⚠️ Troubleshooting

### GPU not detected
```bash
# Check drivers on host
nvidia-smi

# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

### Out of memory
- Check if GPU has 60GB+ VRAM
- Close other processes using GPU
- Adjust `shm_size` in docker-compose

### Model download fails
```bash
# Try with alternative mirror
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download tencent/HunyuanWorld-Voyager --local-dir ./
```

### Permission errors
```bash
# Fix volume permissions
sudo chown -R 1000:1000 /var/lib/docker/volumes/voyager_*
```

### Interface doesn't load
```bash
# Check if port 3500 is free
netstat -tulpn | grep 3500

# Check logs
docker logs hunyuanworld-voyager
```

## 📊 Required Resources

- **VRAM**: Minimum 60GB (RTX A6000, A100, H100)
- **RAM**: 32GB+ recommended
- **Disk**: 500GB+ for models and results
- **CPU**: AVX2 support

## 🔗 Useful Links

- [Original Repository](https://github.com/dmax101/HunyuanWorld-Voyager)
- [HuggingFace Models](https://huggingface.co/tencent/HunyuanWorld-Voyager)
- [TrueNAS Documentation](https://www.truenas.com/docs/)
- [Dockge](https://github.com/louislam/dockge)

## ⚠️ Important Warnings

1. **High Resource Consumption**: This model requires very powerful hardware
2. **Large Download**: Models occupy 100GB+ space
3. **Processing Time**: Video generation can take several minutes
4. **Energy Costs**: High-performance GPUs consume significant power

## 📝 Version Notes

- **v1.0**: Initial configuration for TrueNAS/Dockge
- **CUDA**: 12.4 with full support for RTX 40xx/A100/H100
- **PyTorch**: 2.4.0 optimized for performance

## 🤝 Contributions

For improvements to this Docker configuration, open issues or PRs in the repository.

---

**🌐 Final Access**: After complete configuration, access: `http://[TRUENAS_IP]:3500`

**💡 Tip**: On first run, let the container run for at least 30 minutes to complete all installations before trying to access the web interface.