# The offical impl of RDNet: Reversible Decoupling Network for Single Image Reflection Removal
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-real20)](https://paperswithcode.com/sota/reflection-removal-on-real20?p=reversible-decoupling-network-for-single)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reversible-decoupling-network-for-single/reflection-removal-on-sir-2-objects)](https://paperswithcode.com/sota/reflection-removal-on-sir-2-objects?p=reversible-decoupling-network-for-single)

# Requirements

```
pip install torch>=2.0 torchvision
pip install einops ema-pytorch fsspec fvcore huggingface-hub matplotlib numpy opencv-python omegaconf pytorch-msssim scikit-image scikit-learn scipy tensorboard tensorboardx wandb

```

# Testing 

```python
python3 test_sirs.py --icnn_path <path to the checkpoint> --resume
```
