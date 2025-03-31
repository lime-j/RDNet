<p align="center">
<img src="https://github.com/lime-j/RDNet/blob/main/figures/Title.png?raw=true" width=95%>
<p>

# Reversible Decoupling Network for Single Image Reflection Removal

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-sir-2-objects)](https://paperswithcode.com/sota/reflection-removal-on-sir-2-objects?p=reversible-decoupling-network-for-single)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-sir-2-wild)](https://paperswithcode.com/sota/reflection-removal-on-sir-2-wild?p=reversible-decoupling-network-for-single)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-sir-2-postcard)](https://paperswithcode.com/sota/reflection-removal-on-sir-2-postcard?p=reversible-decoupling-network-for-single)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-nature)](https://paperswithcode.com/sota/reflection-removal-on-nature?p=reversible-decoupling-network-for-single)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-real20)](https://paperswithcode.com/sota/reflection-removal-on-real20?p=reversible-decoupling-network-for-single)

</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2410.08063"> Reversible Decoupling Network for Single Image Reflection Removal</a>
</p>
<p align="center">
  <a href="https://github.com/WHiTEWoLFJ"> Hao Zhao</a> ⚔️,
  <a href="https://github.com/lime-j"> Mingjia Li</a> ⚔️,
  <a href="https://github.com/mingcv"> Qiming Hu</a>,
  <a href="https://sites.google.com/view/xjguo"> Xiaojie Guo</a> 🦅,
  <p align="center">(⚔️: equal contribution, 🦅 : corresponding author)</p>
</p>

<p align="center">
<img src="https://github.com/lime-j/RDNet/blob/main/figures/net.png?raw=true" width=95%>
<p>

<details>
  <summary>Click for the Abstract of RDNet</summary>
  We present a Reversible Decoupling Network (RDNet), which employs a reversible encoder to secure valuable information while flexibly decoupling transmission-and-reflection-relevant features during the forward pass. Furthermore, we customize a transmission-rate-aware prompt generator to dynamically calibrate features, further boosting performance. Extensive experiments demonstrate the superiority of RDNet over existing SOTA methods on five widely-adopted benchmark datasets.
</details>

**Our work is accepted by CVPR 2025! See you at the conference!**

**Our work RDNet wins the NTIRE 2025 Single Image Reflection Removal in the Wild Challenge!**

## 🌠 Gallery


<table class="center">
    <tr>
      <td><p style="text-align: center">Class Room</p></td>
      <td><p style="text-align: center">White Wall Chamber</p></td>
    </tr>
    <tr>
      <td>
        <div style="width: 100%; max-width: 600px; position: relative;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Input_class.png?raw=true" style="width: 100%; height: 300px; display: block;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Ours_class.png?raw=true" style="width: 100%; height: 300px; display: block; position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 0.5s;" onmouseover="this.style.opacity=1;" onmouseout="this.style.opacity=0;">
        </div>
      </td>
      <td>
        <div style="width: 100%; max-width: 600px; position: relative;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/input_white.jpg?raw=true" style="width: 100%; height: 300px; display: block;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Ours_white.png?raw=true" style="width: 100%; height: 300px; display: block; position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 0.5s;" onmouseover="this.style.opacity=1;" onmouseout="this.style.opacity=0;">
        </div>
      </td>
    </tr>
    <tr>
      <td><p style="text-align: center">Car Window</p></td>
      <td><p style="text-align: center">Very Green Office</p></td>
    </tr>
    <tr>
      <td>
        <div style="width: 100%; max-width: 600px; position: relative;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Input_car.jpg?raw=true" style="width: 100%; height: 300px; display: block;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Ours_car.png?raw=true" style="width: 100%; height: 300px; display: block; position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 0.5s;" onmouseover="this.style.opacity=1;" onmouseout="this.style.opacity=0;">
        </div>
      </td>
      <td>
        <div style="width: 100%; max-width: 600px; position: relative;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Input_green.png?raw=true" style="width: 100%; height: 300px; display: block;">
          <img src="https://github.com/lime-j/RDNet/blob/main/figures/Ours_green.png?raw=true" style="width: 100%; height: 300px; display: block; position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 0.5s;" onmouseover="this.style.opacity=1;" onmouseout="this.style.opacity=0;">
        </div>
      </td>
    </tr>
</table>

## Requirements
We recommend torch 2.x for our code, but it should works fine with most of the modern versions.

```
pip install torch>=2.0 torchvision
pip install einops ema-pytorch fsspec fvcore huggingface-hub matplotlib numpy opencv-python omegaconf pytorch-msssim scikit-image scikit-learn scipy tensorboard tensorboardx wandb timm
```

# Testing 
The checkpoint for the main network is available at https://checkpoints.mingjia.li/rdnet.pth ; while the model for cls_model is at https://checkpoints.mingjia.li/cls_model.pth . Please put the cls_model.pth under "pretrained" folder.

```python
python3 test_sirs.py --icnn_path <path to the main checkpoint> --resume
```
# Training 

The training script / data preprocessing script is released. Just use ``train.py`` to train the model.

```python
python3 train.py 
```

# Acknowledgement

We are grateful for the computational resource support provided by Google's TPU Research Cloud and DataCanvas Limited.
