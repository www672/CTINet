# CTI-Net: Cycle-Trend Interactions Network for Lightweight Time Series Forecasting


## Introduction

CTI-Net is a lightweight time series forecasting framework that explicitly models the interaction between cycle and trend components. By decomposing time series into cycle and trend representations and using cycle-based modulation to adaptively adjust trend information, CTI-Net captures the dependence between periodic fluctuations and evolving trend levels while maintaining low computational complexity. 

<p align="center">
<img src=".\images\FIGURE 1.png" width = "700" alt="" align=center />
<br><br>
<b>FIGURE 1.</b> Overall architecture of the CTI-Net framework.
</p>

## Environment Requirements

To get started, ensure you have Conda installed on your system and follow these steps to set up the environment:

```
conda create -n CTINet python=3.10
conda activate CTINet
pip install -r requirements.txt
```

## Data Preparation

* Pre-processed datasets can be downloaded from the following
  links, [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/e1ccfff39ad541908bae/)
  or [Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy?usp=sharing).
* Place the downloaded datasets into the `dataset/` folder, e.g. `dataset/ETTh1.csv`.

## Usage

1. Install the required dependencies.
2. Download data as above, and place them in the folder, `dataset/`.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`,
   e.g. `./scripts/ETTh1.sh`. You might have to change permissions on the script files by running`chmod u+x scripts/*`.

## Acknowledgement

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases and datasets:

* https://github.com/thuml/iTransformer

* https://github.com/salesforce/ETSformer

* https://github.com/yuqinie98/patchtst

* https://github.com/cure-lab/LTSF-Linear

* https://github.com/zhouhaoyi/Informer2020

* https://github.com/thuml/Autoformer

* https://github.com/MAZiqing/FEDformer

* https://github.com/ACAT-SCUT/CycleNet

* https://github.com/ts-kim/RevIN

* https://github.com/timeseriesAI/tsai

