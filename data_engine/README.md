This is the data engine for processing training data of Voyager.


## 🛠️ Dependencies and Installation

Begin by cloning required repositories:
```shell
# VGGT
git clone https://github.com/facebookresearch/vggt.git
touch vggt/vggt/__init__.py  # create an empty init.py

# MoGe
git clone https://github.com/microsoft/MoGe.git

# Metric3D
git clone https://github.com/YvanYin/Metric3D.git
# comment out line 8-12 in Metric3D/mono/utils/comm.py
# add from mono.model.backbones import * to Metric3D/mono/utils/comm.py
```

Install required dependencies:

```shell
conda create -n data_engine python=3.10
conda activate data_engine
pip install -r requirements.txt
```

## 🔑 Run Data Engine

We provide a script to run the data engine.
```shell
bash run.sh
```

# 1. install environment
# project path
cd voyager_dataEngine

# git clone VGGT
https://github.com/facebookresearch/vggt.git
touch vggt/vggt/__init__.py

# git clone MoGe
git clone https://github.com/microsoft/MoGe.git

# git clone Metric3Dv2
git clone https://github.com/YvanYin/Metric3D.git
注释 voyager_dataEngine/Metric3D/mono/utils/comm.py  8-12行
加上from mono.model.backbones import *

# pip install environment
conda create -n voyager_dataengine python=3.10
conda activate voyager_dataengine
pip install -r requirements.txt

# run dataEngine
bash dataEngine.sh