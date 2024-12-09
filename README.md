# RDNet for Single Image Reflection Removal

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
<img src="https://github.com/lime-j/RDNet/blob/main/figures/net.png?raw=true" width=95%>
<p>

We present a Reversible Decoupling Network (RDNet), which employs a reversible encoder to secure valuable information while flexibly decoupling transmission-and-reflection-relevant features during the forward pass. Furthermore, we customize a transmission-rate-aware prompt generator to dynamically calibrate features, further boosting performance. Extensive experiments demonstrate the superiority of RDNet over existing SOTA methods on five widely-adopted benchmark datasets.

# Requirements

```
pip install torch>=2.0 torchvision
pip install einops ema-pytorch fsspec fvcore huggingface-hub matplotlib numpy opencv-python omegaconf pytorch-msssim scikit-image scikit-learn scipy tensorboard tensorboardx wandb timm
```

# Testing 

```python
python3 test_sirs.py --icnn_path <path to the checkpoint> --resume
```
