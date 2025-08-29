<!-- ## **HunyuanVideo** -->

[中文阅读](./README_zh.md)

<!-- <p align="center">
  <img src="./assets/logo.png"  height=100>
</p> -->

# **HunyuanWorld-Voyager**

<div align="center">
  <a href=""><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2506.04225"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=HunyuanWorld-Voyager&message=HuggingFace&color=yellow"></a>
</div>

-----

We introduce HunyuanWorld-Voyager, a novel video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image with user-defined camera path. Voyager can generate 3D-consistent scene videos for world exploration following custom camera trajectories. It can also jointly generate aligned depth and RGB video for effective and direct 3D reconstruction.



## 🔥🔥🔥 News!!
* Sep 2, 2025: 👋 We release the inference code and model weights of HunyuanWorld-Voyager. [Download](./ckpts/README.md).



## 🎥 Demo
### Demo Video
<div align="center">
  <video src="" width="80%" poster=""> </video>
</div>

### Camera-Controllable Video Generation 
|  Input | Generated Video  |
|:----------------:|:----------------:|
|  <img src="assets/demo/camera/input1.png" width="80%">   |       <video src="https://github.com/user-attachments/assets/2b03ecd5-9a8f-455c-bf04-c668d3a61b04" width="100%"> </video> |
| <img src="assets/demo/camera/input2.png" width="80%">         |       <video src="https://github.com/user-attachments/assets/45844ac0-c65a-4e04-9f7d-4c72d47e0339" width="100%"> </video>        | 
| <img src="assets/demo/camera/input3.png" width="80%">         |       <video src="https://github.com/user-attachments/assets/f7f48473-3bb5-4a30-bd22-af3ca95ee8dc" width="100%"> </video>        |

### Multiple Applications

- Video Reconstruction

| Generated Video | Reconstructed Point Cloud |
|:---------------:|:--------------------------------:|
| <video src="https://github.com/user-attachments/assets/72a41804-63fc-4596-963d-1497e68f7790" width="100%"> </video> | <video src="assets/demo/reconstruction/recording.mp4" width="100%"> </video> |

- Image-to-3D Generation

| | |
|:---------------:|:---------------:|
| <video src="https://github.com/user-attachments/assets/886aa86d-990e-4b86-97a5-0b9110862d14" width="100%"> </video> | <video src="https://github.com/user-attachments/assets/4c1734ba-4e78-4979-b30e-3c8c97aa984b" width="100%"> </video> |

- Video Depth Estimation

| | |
|:---------------:|:---------------:|
| <video src="https://github.com/user-attachments/assets/e4c8b729-e880-4be3-826f-429a5c1f12cd" width="100%"> </video> | <video src="https://github.com/user-attachments/assets/7ede0745-cde7-42f1-9c28-e4dca90dac52" width="100%"> </video> |


## ☯️ **HunyuanWorld-Voyager Introduction**

###  Architecture

Voyager consists of two key components:

(1) World-Consistent Video Diffusion: A unified architecture that jointly generates aligned RGB and depth video sequences, conditioned on existing world observation to ensure global coherence.

(2) Long-Range World Exploration: An efficient world cache with point culling and an auto-regressive inference with smooth video sampling for iterative scene extension with context-aware consistency.

To train Voyager, we propose a scalable data engine, i.e., a video reconstruction pipeline that automates camera pose estimation and metric depth prediction for arbitrary videos, enabling large-scale, diverse training data curation without manual 3D annotations. Using this pipeline, we compile a dataset of over 100,000 video clips, combining real-world captures and synthetic Unreal Engine renders.
<p align="center">
  <img src="./assets/backbone.jpg"  height=500>
</p>


### Performance

<table class="comparison-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>WorldScore Average</th>
      <th>Camera Control</th>
      <th>Object Control</th>
      <th>Content Alignment</th>
      <th>3D Consistency</th>
      <th>Photometric Consistency</th>
      <th>Style Consistency</th>
      <th>Subjective Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>WonderJourney</td>
      <td><u>63.75</u></td>
      <td><u>84.6</u></td>
      <td>37.1</td>
      <td>35.54</td>
      <td>80.6</td>
      <td>79.03</td>
      <td>62.82</td>
      <td><b>66.56</b></td>
    </tr>
    <tr>
      <td>WonderWorld</td>
      <td><b>72.69</b></td>
      <td><b><u>92.98</u></b></td>
      <td>51.76</td>
      <td><b><u>71.25</u></b></td>
      <td><b><u>86.87</u></b></td>
      <td>85.56</td>
      <td>70.57</td>
      <td>49.81</td>
    </tr>
    <tr>
      <td>EasyAnimate</td>
      <td>52.85</td>
      <td>26.72</td>
      <td>54.5</td>
      <td>50.76</td>
      <td>67.29</td>
      <td>47.35</td>
      <td><u>73.05</u></td>
      <td>50.31</td>
    </tr>
    <tr>
      <td>Allegro</td>
      <td>55.31</td>
      <td>24.84</td>
      <td><u>57.47</u></td>
      <td><u>51.48</u></td>
      <td>70.5</td>
      <td>69.89</td>
      <td>65.6</td>
      <td>47.41</td>
    </tr>
    <tr>
      <td>Gen-3</td>
      <td>60.71</td>
      <td>29.47</td>
      <td><b>62.92</b></td>
      <td>50.49</td>
      <td>68.31</td>
      <td><b>87.09</b></td>
      <td>62.82</td>
      <td><u>63.85</u></td>
    </tr>
    <tr>
      <td>CogVideoX-I2V</td>
      <td>62.15</td>
      <td>38.27</td>
      <td>40.07</td>
      <td>36.73</td>
      <td><b>86.21</b></td>
      <td><b><u>88.12</u></b></td>
      <td><b>83.22</b></td>
      <td>62.44</td>
    </tr>
    <tr class="voyager-row">
      <td><b>Voyager</b></td>
      <td><b><u>77.62</u></b></td>
      <td><b>85.95</b></td>
      <td><b><u>66.92</u></b></td>
      <td><b>68.92</b></td>
      <td><u>81.56</u></td>
      <td><u>85.99</u></td>
      <td><b><u>84.89</u></b></td>
      <td><b><u>71.09</u></b></td>
    </tr>
  </tbody>
  <caption>Quantitative comparison on <i>WorldScore Benchmark</i>. <b><u>Bold and underline</u></b> indicates the 1st, <b>Bold</b> indicates the 2nd, <u>underline</u> indicates the 3rd.</caption>
</table>



## 📜 Requirements

The following table shows the requirements for running Voyager (batch size = 1) to generate videos:

|      Model       | Resolution  | GPU Peak Memory  |
|:----------------:|:-----------:|:----------------:|
| HunyuanWorld-Voyager |    540p     |       60GB        |


* An NVIDIA GPU with CUDA support is required. 
  * The model is tested on a single 80G GPU.
  * **Minimum**: The minimum GPU memory required is 60GB for 540p.
  * **Recommended**: We recommend using a GPU with 80GB of memory for better generation quality.
* Tested operating system: Linux

## 🛠️ Dependencies and Installation

Begin by cloning the repository:
```shell
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
cd HunyuanWorld-Voyager
```

### Installation Guide for Linux

We recommend CUDA versions 12.4 or 11.8 for the manual installation.

```shell
# 1. Create conda environment
conda create -n voyager python==3.11.9

# 2. Activate the environment
conda activate voyager

# 3. Install PyTorch and other dependencies using conda
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r requirements.txt
python -m pip install transformers==4.39.3

# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install flash-attn

# 6. Install xDiT for parallel inference (It is recommended to use torch 2.4.0 and flash-attn 2.6.3)
python -m pip install xfuser==0.4.2
```

In case of running into float point exception(core dump) on the specific GPU type, you may try the following solutions:

```shell
# Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/
```

## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](ckpts/README.md).



## 🔑 Inference

### Create Input Condition
```bash
cd data_engine

python3 create_input.py --image_path "your_input_image" --render_output_dir "demo/example/" --type "forward"
```
We provide the following types of camera path:
- forward
- backward
- left
- right
- turn_left
- turn_right
You can also modify the camera path in the `create_input.py` file.

### Single-GPU Inference
```bash
cd HunyuanWorld-Voyager

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "demo/example" \
    --prompt "Waterfall at Sunset, waterfall, grassy landscape." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results
```
You can add "--use-context-block" to add the context block in the inference.

### Parallel Inference on Multiple GPUs by xDiT

[xDiT](https://github.com/xdit-project/xDiT) is a Scalable Inference Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters.
It has successfully provided low-latency parallel inference solutions for a variety of DiTs models, including mochi-1, CogVideoX, Flux.1, SD3, etc. This repo adopted the [Unified Sequence Parallelism (USP)](https://arxiv.org/abs/2405.07719) APIs for parallel inference of the HunyuanVideo-I2V model.

For example, to generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanWorld-Voyager

ALLOW_RESIZE_FOR_SP=1 torchrun --nproc_per_node=8 \
    sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "demo/example" \
    --prompt "Waterfall at Sunset, waterfall, grassy landscape." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --save-path ./results \
    --ulysses-degree 8 \
    --ring-degree 1
```

The number of GPUs equals the product of `--ulysses-degree` and `--ring-degree.` Feel free to adjust these parallel configurations to optimize performance.


<p align="center">
<table align="center">
<thead>
<tr>
    <th colspan="4">Latency (Sec) for 512x768 (49 frames 50 steps) on 8 x H20 GPU</th>
</tr>
<tr>
    <th>1</th>
    <th>2</th>
    <th>4</th>
    <th>8</th>
</tr>
</thead>
<tbody>
<tr>
    <th>1925</th>
    <th>1018 (1.89x)</th>
    <th>534 (3.60x)</th>
    <th>288 (6.69x)</th>
</tr>

</tbody>
</table>
</p>


## ⚙️ Data Engine

We also release the data engine of HunyuanWorld-Voyager, which can be used to generate scalable data for RGB-D video training. Please refer to [data_engine](./data_engine/README.md) for more details.

<p align="center">
  <img src="./assets/data_engine.jpg"  height=500>
</p>

## 🔗 BibTeX

If you find [Voyager](https://arxiv.org/abs/2506.04225) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{huang2025voyager,
  title={Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation},
  author={Huang, Tianyu and Zheng, Wangguandong and Wang, Tengfei and Liu, Yuhao and Wang, Zhenwei and Wu, Junta and Jiang, Jie and Li, Hui and Lau, Rynson WH and Zuo, Wangmeng and Guo, Chunchao},
  journal={arXiv preprint arXiv:2506.04225},
  year={2025}
}
```



## Acknowledgements

We would like to thank [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) and [HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0). We also thank [VGGT](https://github.com/facebookresearch/vggt), [MoGE](https://github.com/microsoft/MoGe), [Metric3D](https://github.com/YvanYin/Metric3D), for their open research and exploration.

