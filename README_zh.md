[English](README.md)

# **HunyuanWorld-Voyager**

<p align="center">
  <img src="assets/teaser_zh.png">
</p>

<div align="center">
  <a href="https://3d-models.hunyuan.tencent.com/world/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a> &ensp;
  <a href="https://arxiv.org/abs/2506.04225"><img src="https://img.shields.io/static/v1?label=Tech%20Report&message=Arxiv&color=red"></a> &ensp;
  <a href="https://huggingface.co/tencent/HunyuanWorld-Voyager"><img src="https://img.shields.io/static/v1?label=HunyuanWorld-Voyager&message=HuggingFace&color=yellow"></a>
</div>

-----

我们提出了混元世界模型-Voyager，一个新颖的视频扩散框架，能够从单张图像生成世界一致的3D点云序列，并跟随用户定义的相机路径进行世界探索。Voyager还能够联合生成对齐的深度和RGB视频，实现有效的3D重建。


## 🔥🔥🔥 最新消息!!
* Sep 2, 2025: 👋 我们发布了HunyuanWorld-Voyager的推理代码和模型权重。[下载](ckpts/README.md).


## 🎥 演示
### 演示视频

<div align="center">
  <video src="https://github.com/user-attachments/assets/d095a4fd-22a6-41c6-bedd-3e45b468eb98" width="80%" poster=""> </video>
</div>

### 相机可控视频生成

|  输入 | 生成视频  |
|:----------------:|:----------------:|
|  <img src="assets/demo/camera/input1.png" width="80%">        |       <video src="https://github.com/user-attachments/assets/2b03ecd5-9a8f-455c-bf04-c668d3a61b04" width="100%"> </video>        |
| <img src="assets/demo/camera/input2.png" width="80%">         |       <video src="https://github.com/user-attachments/assets/45844ac0-c65a-4e04-9f7d-4c72d47e0339" width="100%"> </video>        | 
| <img src="assets/demo/camera/input3.png" width="80%">         |       <video src="https://github.com/user-attachments/assets/f7f48473-3bb5-4a30-bd22-af3ca95ee8dc" width="100%"> </video>        |

### 多样化应用

- 视频重建

| 生成视频 | 重建点云 |
|:---------------:|:--------------------------------:|
| <video src="https://github.com/user-attachments/assets/72a41804-63fc-4596-963d-1497e68f7790" width="100%"> </video> | <video src="https://github.com/user-attachments/assets/67574e9c-9e21-4ed6-9503-e65d187086a2" width="100%"> </video> |

- 图像到3D生成

| | |
|:---------------:|:---------------:|
| <video src="https://github.com/user-attachments/assets/886aa86d-990e-4b86-97a5-0b9110862d14" width="100%"> </video> | <video src="https://github.com/user-attachments/assets/4c1734ba-4e78-4979-b30e-3c8c97aa984b" width="100%"> </video> |

- 视频深度估计

| | |
|:---------------:|:---------------:|
| <video src="https://github.com/user-attachments/assets/e4c8b729-e880-4be3-826f-429a5c1f12cd" width="100%"> </video> | <video src="https://github.com/user-attachments/assets/7ede0745-cde7-42f1-9c28-e4dca90dac52" width="100%"> </video> |


## ☯️ **混元世界模型-Voyager 介绍**
### 架构

Voyager 由两个关键组件组成：

(1) 世界一致的视频扩散：一个统一的架构，联合生成对齐的RGB和深度视频序列，基于现有的世界观察，确保全局一致性。

(2) 长距离世界探索：一个高效的点云剔除和自回归推理的缓存，用于迭代场景扩展，并使用上下文感知的连续性进行平滑的视频采样。

为了训练Voyager，我们提出了一种可扩展的数据引擎，即一个视频重建管道，自动估计相机位姿和度量深度，用于任意视频，实现了大规模、多样化的训练数据收集，无需手动3D标注。使用这个管道，我们编译了一个超过100,000个视频片段的数据集，结合了真实世界的捕捉和合成Unreal Engine渲染。

<p align="center">
  <img src="assets/backbone.jpg"  height=500>
</p>

### 性能

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
      <td>🟡63.75</td>
      <td>🟡84.6</td>
      <td>37.1</td>
      <td>35.54</td>
      <td>80.6</td>
      <td>79.03</td>
      <td>62.82</td>
      <td>🟢66.56</td>
    </tr>
    <tr>
      <td>WonderWorld</td>
      <td>🟢72.69</td>
      <td>🔴92.98</td>
      <td>51.76</td>
      <td>🔴71.25</td>
      <td>🔴86.87</td>
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
      <td>🟡73.05</td>
      <td>50.31</td>
    </tr>
    <tr>
      <td>Allegro</td>
      <td>55.31</td>
      <td>24.84</td>
      <td>🟡57.47</td>
      <td>🟡51.48</td>
      <td>70.5</td>
      <td>69.89</td>
      <td>65.6</td>
      <td>47.41</td>
    </tr>
    <tr>
      <td>Gen-3</td>
      <td>60.71</td>
      <td>29.47</td>
      <td>🟢62.92</td>
      <td>50.49</td>
      <td>68.31</td>
      <td>🟢87.09</td>
      <td>62.82</td>
      <td>🟡63.85</td>
    </tr>
    <tr>
      <td>CogVideoX-I2V</td>
      <td>62.15</td>
      <td>38.27</td>
      <td>40.07</td>
      <td>36.73</td>
      <td>🟢86.21</td>
      <td>🔴88.12</td>
      <td>🟢83.22</td>
      <td>62.44</td>
    </tr>
    <tr class="voyager-row">
      <td><b>Voyager</b></td>
      <td>🔴77.62</td>
      <td>🟢85.95</td>
      <td>🔴66.92</td>
      <td>🟢68.92</td>
      <td>🟡81.56</td>
      <td>🟡85.99</td>
      <td>🔴84.89</td>
      <td>🔴71.09</td>
    </tr>
  </tbody>
  <caption><i>WorldScore Benchmark</i>的定量比较结果. 🔴 表示第1名, 🟢 表示第2名, 🟡 表示第3名.</caption>
</table>


## 📜 要求

以下表格展示了运行Voyager（批量大小 = 1）生成视频的要求：

|      模型       | 分辨率  | GPU 峰值内存  |
|:----------------:|:-----------:|:----------------:|
| 混元世界模型-Voyager |    540p     |       60GB        |

* 需要NVIDIA GPU支持CUDA。
  * 模型在单个80G GPU上测试。
  * **最小值**: 最小GPU内存要求为540p的60GB。
  * **推荐**: 我们推荐使用80GB内存的GPU以获得更好的生成质量。
* 测试操作系统: Linux


## 🛠️ 依赖和安装

首先克隆仓库：
```shell
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager
cd HunyuanWorld-Voyager
```

### Linux 安装指南

我们推荐CUDA版本12.4或11.8进行手动安装。

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

在特定GPU类型上运行时，如果出现浮点异常（core dump），您可以尝试以下解决方案：

```shell
# Making sure you have installed CUDA 12.4, CUBLAS>=12.4.5.8, and CUDNN>=9.00 (or simply using our CUDA 12 docker image).
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/
```


## 🧱 下载预训练模型

下载预训练模型的详细信息请参考[这里](ckpts/README.md)。


## 🔑 推理

### 创建输入条件

```bash
cd data_engine

python3 create_input.py --image_path "your_input_image" --render_output_dir "examples/case/" --type "forward"
```
我们提供了以下类型的相机路径：
- forward
- backward
- left
- right
- turn_left
- turn_right
您也可以在`create_input.py`文件中修改相机路径。

### 单GPU推理

```bash
cd HunyuanWorld-Voyager

python3 sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "An old-fashioned European village with thatched roofs on the houses." \
    --i2v-stability \
    --infer-steps 50 \
    --flow-reverse \
    --flow-shift 7.0 \
    --seed 0 \
    --embedded-cfg-scale 6.0 \
    --use-cpu-offload \
    --save-path ./results
```
您可以添加"--use-context-block"来添加推理中的上下文块。

### 多GPU并行推理

[xDiT](https://github.com/xdit-project/xDiT) 是一个可扩展的推理引擎，用于多GPU集群上的扩散Transformer（DiTs）。
它成功地为各种DiTs模型（包括mochi-1、CogVideoX、Flux.1、SD3等）提供了低延迟的并行推理解决方案。这个仓库采用了[统一序列并行（USP）](https://arxiv.org/abs/2405.07719) API来并行推理HunyuanVideo-I2V模型。

例如，要使用8个GPU生成视频，您可以使用以下命令：

```bash
cd HunyuanWorld-Voyager

ALLOW_RESIZE_FOR_SP=1 torchrun --nproc_per_node=8 \
    sample_image2video.py \
    --model HYVideo-T/2 \
    --input-path "examples/case1" \
    --prompt "An old-fashioned European village with thatched roofs on the houses." \
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

GPU数量等于`--ulysses-degree`和`--ring-degree`的乘积。您可以自由调整这些并行配置以优化性能。

<p align="center">
<table align="center">
<thead>
<tr>
    <th colspan="4">512x768（49帧，50步）在8 x H20 GPU上的延迟（秒）</th>
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


## ⚙️ 数据引擎

我们发布了混元世界模型-Voyager的数据引擎，可以用于生成可扩展的RGB-D视频训练数据。请参考[data_engine](data_engine/README.md)了解更多细节。

<p align="center">
  <img src="assets/data_engine.jpg"  height=500>
</p>


## 🔗 引用

如果您发现[Voyager](https://arxiv.org/abs/2506.04225)对您的研究或应用有用，请使用以下BibTeX引用：

```BibTeX
@article{huang2025voyager,
  title={Voyager: Long-Range and World-Consistent Video Diffusion for Explorable 3D Scene Generation},
  author={Huang, Tianyu and Zheng, Wangguandong and Wang, Tengfei and Liu, Yuhao and Wang, Zhenwei and Wu, Junta and Jiang, Jie and Li, Hui and Lau, Rynson WH and Zuo, Wangmeng and Guo, Chunchao},
  journal={arXiv preprint arXiv:2506.04225},
  year={2025}
}
```


## 致谢

我们感谢[HunyuanWorld](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)、[Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)和[HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V)。我们也感谢[VGGT](https://github.com/facebookresearch/vggt)、[MoGE](https://github.com/microsoft/MoGe)、[Metric3D](https://github.com/YvanYin/Metric3D)的贡献者。
