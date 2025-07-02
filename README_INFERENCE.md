# RDNet Custom Image Inference

This guide explains how to run inference using the pretrained RDNet model on your custom images with reflections.

## Prerequisites

Make sure you have:
1. The pretrained model file: `pretrained/rdnet.pth` 
2. A folder containing your input images with reflections
3. Python environment with the required dependencies

## Usage

We provide two inference scripts:

### Option 1: Simple Script (Recommended)

```bash
cd RDNet
python inference_simple.py /path/to/your/images
```

With custom output directory:
```bash
python inference_simple.py /path/to/your/images ./my_results
```

With custom model and output:
```bash
python inference_simple.py /path/to/your/images ./my_results pretrained/rdnet.pth 0
```


#### Parameters for Advanced Script

- `--input_dir`: Directory containing your input images (required)
- `--output_dir`: Directory where results will be saved (default: `./inference_results`)
- `--model_path`: Path to the pretrained model (default: `pretrained/rdnet.pth`)
- `--gpu`: GPU ID to use (default: '0')

## Output Structure

For each input image, the script will create a folder with the following files:

```
output_dir/
├── image1_name/
│   ├── ytmt_ucs_sirs_l.png    # Clean transmission layer (main result)
│   ├── ytmt_ucs_sirs_r.png    # Reflection layer
│   └── m_input.png            # Original input image
├── image2_name/
│   ├── ytmt_ucs_sirs_l.png
│   ├── ytmt_ucs_sirs_r.png
│   └── m_input.png
└── ...
```

The `*_l.png` file is your main result - the clean image with reflections removed.

## Supported Image Formats

The script supports common image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`

## Example

```bash
# If your images are in a folder called "my_photos"
python inference_simple.py ./my_photos ./cleaned_photos

# The results will be saved in ./cleaned_photos/
```

## Notes

- Images are automatically resized to be divisible by 32 for model compatibility
- The script processes one image at a time to avoid memory issues
- GPU is used by default if available, otherwise CPU will be used 