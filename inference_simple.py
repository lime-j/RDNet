import os
import sys
from os.path import join
import torch.backends.cudnn as cudnn

import data.dataset_sir as datasets
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils


def run_inference(input_dir, output_dir='./inference_results', model_path='pretrained/rdnet.pth', gpu_id='0'):
    """
    Run inference on custom images using RDNet model
    
    Args:
        input_dir: Directory containing input images with reflections
        output_dir: Directory to save output images  
        model_path: Path to pretrained model
        gpu_id: GPU ID to use
    """
    
    # Validate input directory
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    # Validate model path
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear sys.argv to avoid argument conflicts
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    try:
        # Set up options
        opt = TrainOptions().parse()
        
        # Override options for inference
        opt.isTrain = False
        opt.icnn_path = model_path  # Set path to our pretrained model
        opt.resume = True  # Enable loading from checkpoint
        # Convert gpu string to list format expected by the model
        opt.gpu_ids = [int(gpu_id)] if gpu_id != '-1' else []
        opt.no_log = True
        opt.display_id = 0
        opt.verbose = False
        opt.if_align = True  # Enable image alignment
        opt.nThreads = 1
        
        cudnn.benchmark = True
        
        # Create dataset for custom images
        custom_dataset = datasets.RealDataset(input_dir)
        
        # Create dataloader
        custom_dataloader = datasets.DataLoader(
            custom_dataset, batch_size=1, shuffle=False,
            num_workers=opt.nThreads, pin_memory=True)
        
        # Create dummy datasets for Engine initialization (required by Engine constructor)
        # These won't be used, but Engine expects them
        dummy_dataset = datasets.RealDataset(input_dir, size=1)
        
        # Initialize Engine
        engine = Engine(opt, dummy_dataset, dummy_dataset, dummy_dataset, dummy_dataset)
        
        print(f"Starting inference on {len(custom_dataset)} images...")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Model path: {model_path}")
        
        # Run inference
        print("Running inference...")
        for i, data in enumerate(custom_dataloader):
            print(f"Processing image {i+1}/{len(custom_dataset)}: {data['fn'][0]}")
            engine.model.test(data, savedir=output_dir)
        
        print(f"Inference completed! Results saved to {output_dir}")
        print(f"For each input image, you'll find:")
        print(f"  - {{name}}/{{model_name}}_l.png: Clean transmission layer (main result)")
        print(f"  - {{name}}/{{model_name}}_r.png: Reflection layer")
        print(f"  - {{name}}/m_input.png: Original input image")
        
    finally:
        # Restore original argv
        sys.argv = original_argv


def main():
    """
    Main function with simple argument parsing
    Usage examples:
      python inference_simple.py /path/to/images
      python inference_simple.py /path/to/images /path/to/output
      python inference_simple.py /path/to/images /path/to/output pretrained/rdnet.pth
      python inference_simple.py /path/to/images /path/to/output pretrained/rdnet.pth 0
    """
    if len(sys.argv) < 2:
        print("Usage: python inference_simple.py <input_dir> [output_dir] [model_path] [gpu_id]")
        print("Example: python inference_simple.py ./my_images ./results pretrained/rdnet.pth 0")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './inference_results'
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'pretrained/rdnet.pth' 
    gpu_id = sys.argv[4] if len(sys.argv) > 4 else '0'
    
    run_inference(input_dir, output_dir, model_path, gpu_id)


if __name__ == '__main__':
    main() 