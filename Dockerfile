FROM nvidia/cuda:12.4-devel-ubuntu22.04

# Evitar prompts interativos durante a instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configurar Python 3.11 como padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Instalar pip mais recente
RUN python3 -m pip install --upgrade pip

# Criar diretório de trabalho
WORKDIR /app

# Clonar o repositório HunyuanWorld-Voyager
RUN git clone https://github.com/dmax101/HunyuanWorld-Voyager.git .

# Instalar PyTorch e dependências CUDA
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Instalar dependências do projeto
RUN pip install -r requirements.txt

# Instalar dependências adicionais mencionadas no README
RUN pip install transformers==4.39.3 \
    flash-attn \
    xfuser==0.4.2 \
    nvidia-cublas-cu12==12.4.5.8 \
    scipy==1.11.4

# Instalar dependências para criação de condições de entrada
RUN pip install --no-deps git+https://github.com/microsoft/MoGe.git
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38

# Criar diretórios necessários
RUN mkdir -p /app/ckpts /app/examples /app/results /app/temp /app/temp_images

# Configurar variáveis de ambiente para CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# Expor porta para Gradio
EXPOSE 8080

# Criar script de inicialização
RUN echo '#!/bin/bash\n\
echo "=== HunyuanWorld-Voyager Container Started ==="\n\
echo "GPU Status:"\n\
nvidia-smi || echo "No GPU detected or nvidia-smi not available"\n\
echo ""\n\
echo "Available commands:"\n\
echo "  - python3 app.py                    # Start Gradio web interface"\n\
echo "  - python3 sample_image2video.py     # CLI inference"\n\
echo "  - bash                              # Interactive shell"\n\
echo ""\n\
echo "Web interface will be available at: http://localhost:8080"\n\
echo ""\n\
if [ "$1" = "web" ]; then\n\
    echo "Starting Gradio web interface..."\n\
    python3 app.py\n\
elif [ "$1" = "shell" ]; then\n\
    echo "Starting interactive shell..."\n\
    /bin/bash\n\
else\n\
    echo "Usage: docker run ... [web|shell]"\n\
    echo "Default: Starting Gradio web interface..."\n\
    python3 app.py\n\
fi' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Definir entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["web"]