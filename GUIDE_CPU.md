# 🖥️ HunyuanWorld-Voyager CPU for TrueNAS/Dockge

## ⚠️ CPU-Only Version (No GPU Required)

This is a special version of HunyuanWorld-Voyager that runs **CPU-only**, ideal for NAS systems that don't have CUDA/NVIDIA GPU support.

### 🔍 CPU Version Limitations

- ❌ **No video generation** (requires GPU with 60GB+ VRAM)
- ❌ **No diffusion model processing** (too heavy for CPU)
- ✅ **Functional web interface** for testing
- ✅ **Image upload** and validation
- ✅ **Processing simulation** for demonstration

## 🚀 Installation on Dockge

### 1. Create Stack in Dockge

1. **Open Dockge**: `http://[TRUENAS_IP]:5001`
2. **New stack**: Name `hunyuanworld-voyager-cpu`
3. **Paste the docker-compose below**:

```yaml
version: '3.8'

services:
  hunyuanworld-voyager-cpu:
    image: ubuntu:22.04
    container_name: hunyuanworld-voyager-cpu
    restart: unless-stopped
    
    entrypoint: ["/bin/bash", "-c"]
    command:
      - |
        set -e
        echo "🚀 HunyuanWorld-Voyager CPU - Starting..."
        echo "⚠️  WARNING: Running in CPU-only mode"
        
        export DEBIAN_FRONTEND=noninteractive
        
        # Install dependencies
        apt-get update && apt-get install -y \
            python3.11 python3.11-dev python3-pip git curl \
            libgl1-mesa-glx libglib2.0-0 build-essential
        
        # Configure Python
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        python3 -m pip install --upgrade pip
        
        # Clone repository
        if [ ! -d "/app/.git" ]; then
            git clone https://github.com/dmax101/HunyuanWorld-Voyager.git /app
        fi
        
        cd /app
        
        # Install PyTorch CPU-only
        pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
            --index-url https://download.pytorch.org/whl/cpu
        
        # Install basic dependencies
        pip install gradio transformers pillow numpy opencv-python \
                   imageio scipy loguru tqdm pandas
        
        # Create simplified CPU demo
        cat > /app/app_cpu.py << 'EOF'
import gradio as gr
import numpy as np
from PIL import Image
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def process_demo(image, direction, prompt):
    if image is None:
        return None, "❌ Upload an image first"
    
    time.sleep(2)  # Simulate processing
    
    result = f"""
✅ Interface working!

📷 Image: Received ({image.size})
🧭 Direction: {direction}
📝 Prompt: {prompt}

⚠️ CPU VERSION - Limited functionality
🎯 For complete generation, use NVIDIA GPU 60GB+
    """
    
    return image, result

with gr.Blocks(title="Voyager CPU") as demo:
    gr.Markdown("# ☯️ HunyuanWorld-Voyager (CPU Mode)")
    
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload", type="pil")
            direction = gr.Dropdown(
                ["forward", "backward", "left", "right"],
                label="Direction", value="forward"
            )
            prompt = gr.Textbox(label="Prompt", lines=3)
            btn = gr.Button("🔍 Test", variant="primary")
        
        with gr.Column():
            out_img = gr.Image(label="Output")
            out_txt = gr.Textbox(label="Status", lines=10)
    
    btn.click(process_demo, [image, direction, prompt], [out_img, out_txt])
    
    gr.Markdown("""
    ## 💡 This is a TEST version
    - ✅ Functional interface
    - ❌ No video generation
    - ❌ No AI processing
    """)

demo.launch(server_name="0.0.0.0", server_port=8080, share=False)
EOF
        
        echo "✅ CPU configuration completed!"
        echo "🌐 Access: http://localhost:3500"
        python3 /app/app_cpu.py
    
    ports:
      - "3500:8080"
    
    volumes:
      - voyager_cpu_app:/app
      - voyager_cpu_data:/app/temp
    
    environment:
      - CUDA_VISIBLE_DEVICES=""
      - PYTHONUNBUFFERED=1
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=8080
    
    shm_size: 1gb
    
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080 || exit 1"]
      interval: 60s
      timeout: 10s
      retries: 3
      start_period: 180s

volumes:
  voyager_cpu_app:
  voyager_cpu_data:
```

### 2. Deploy

1. **Click "Deploy"**
2. **Wait 5-10 minutes** (initial build)
3. **Access**: `http://[TRUENAS_IP]:3500`

## 🎯 What Works

### ✅ Available
- Gradio web interface
- Image upload
- Camera direction selection
- Text prompt input
- Input validation
- Visual demonstration

### ❌ Not Available (Requires GPU)
- Video generation
- Depth processing
- Diffusion models
- MoGE inference
- Real AI processing

## 🔧 What the CPU Version is For

1. **Test the interface** before investing in GPU
2. **Validate connectivity** and NAS configuration
3. **Interface demonstration**
4. **Front-end development**
5. **Proof of concept** for stakeholders

## 🚀 Upgrading to GPU

If you decide to use GPU later:

```bash
# Stop CPU version
docker stop hunyuanworld-voyager-cpu

# Deploy GPU version (use docker-compose-dockge.yml)
# Requires: NVIDIA GPU + 60GB VRAM + NVIDIA Container Toolkit
```

## 📊 Required Resources (CPU)

- **CPU**: 2+ cores
- **RAM**: 4GB+ (minimum)
- **Disk**: 5GB (without models)
- **Network**: Port 3500 available

## 🔧 Useful Commands

```bash
# View logs
docker logs -f hunyuanworld-voyager-cpu

# Access container
docker exec -it hunyuanworld-voyager-cpu bash

# Restart
docker restart hunyuanworld-voyager-cpu

# Stop
docker stop hunyuanworld-voyager-cpu
```

## ❓ Troubleshooting

### Interface doesn't load
```bash
# Check if container is running
docker ps | grep voyager-cpu

# Check logs
docker logs hunyuanworld-voyager-cpu

# Check port
netstat -tulpn | grep 3500
```

### Python error
```bash
# Access container and check
docker exec -it hunyuanworld-voyager-cpu bash
python3 --version
pip list
```

### Slow processing
- **Normal**: CPU is much slower than GPU
- **Use only for interface testing**
- **For production**: Migrate to GPU version

## 💡 Next Steps

### If Interface Works:
1. ✅ Your NAS supports Docker correctly
2. ✅ Network is configured
3. ✅ Can consider GPU upgrade

### For Full Functionality:
1. **Acquire NVIDIA GPU** (RTX A6000, A100, H100)
2. **Install NVIDIA Container Toolkit**
3. **Migrate to docker-compose-dockge.yml**
4. **Download models** (100GB+)

## 🔗 Related Links

- [Full GPU Version](docker-compose-dockge.yml)
- [GPU Guide](GUIDE_DOCKGE.md)
- [Repository](https://github.com/dmax101/HunyuanWorld-Voyager)

---

**💡 This CPU version is ideal for testing if your infrastructure works before investing in expensive GPU hardware!**