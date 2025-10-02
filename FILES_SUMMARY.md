# 📁 HunyuanWorld-Voyager Docker Files Summary

## 🐳 Docker Compose Files

### Main Files (English)
- **`docker-compose-dockge.yml`** - Full GPU version for Dockge (NVIDIA GPU required)
- **`docker-compose-cpu.yml`** - CPU-only version for testing (no GPU required)
- **`docker-compose.yml`** - Alternative GPU version with external volumes

### 📖 Documentation (English)
- **`GUIDE_DOCKGE.md`** - Complete installation guide for GPU version on TrueNAS/Dockge
- **`GUIDE_CPU.md`** - Installation guide for CPU-only version

### 🔧 Configuration Files
- **`Dockerfile`** - Custom Docker image build file
- **`.env.example`** - Environment variables template
- **`.dockerignore`** - Docker build ignore patterns

## 🎯 Recommended Usage

### For Production (Full Functionality)
Use: **`docker-compose-dockge.yml`**
- Requires: NVIDIA GPU with 60GB+ VRAM
- Features: Complete video generation
- Guide: `GUIDE_DOCKGE.md`

### For Testing (Interface Only)
Use: **`docker-compose-cpu.yml`**
- Requires: CPU only (any NAS)
- Features: Interface testing
- Guide: `GUIDE_CPU.md`

## 🚀 Quick Start

1. **Choose version** based on your hardware
2. **Open Dockge**: `http://[TRUENAS_IP]:5001`
3. **Create new stack** with chosen docker-compose file
4. **Deploy** and wait for setup
5. **Access**: `http://[TRUENAS_IP]:3500`

## 📋 File Status

✅ **Kept (English versions)**:
- docker-compose-dockge.yml (GPU version)
- docker-compose-cpu.yml (CPU version)
- GUIDE_DOCKGE.md (GPU guide)
- GUIDE_CPU.md (CPU guide)

❌ **Removed (Portuguese duplicates)**:
- docker-compose-dockge-pt.yml
- docker-compose-cpu-simple.yml
- GUIA_DOCKGE.md
- GUIA_CPU.md
- README_DOCKER.md
- setup_truenas.sh

## 🌐 Access Points

- **Web Interface**: `http://[TRUENAS_IP]:3500`
- **Container Name**: `hunyuanworld-voyager` (GPU) or `hunyuanworld-voyager-cpu` (CPU)
- **Repository**: https://github.com/dmax101/HunyuanWorld-Voyager

---

**All files are now in English and ready for international use! 🌟**