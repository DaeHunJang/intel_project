#!/usr/bin/env python3
import os
import sys
import argparse
import glob
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.model.optimizer import ModelOptimizer
from smart_pub.utils.helpers import get_logger
from smart_pub.config import MODEL_ID, MODEL_DIR

logger = get_logger("model_optimizer")

def find_model_files(model_path):
    """Find pytorch_model.bin or safetensors files in the model directory"""
    model_files = []
    
    # Convert to Path object for easier handling
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.error(f"Model directory does not exist: {model_path}")
        return model_files
    
    # Search patterns for model files
    patterns = [
        "pytorch_model.bin",
        "model.safetensors", 
        "pytorch_model*.bin",
        "model*.safetensors",
        "model-*.safetensors"
    ]
    
    logger.info(f"Searching for model files in: {model_path}")
    
    for pattern in patterns:
        files = list(model_path.glob(pattern))
        if files:
            model_files.extend(files)
            logger.info(f"Found {len(files)} files matching pattern '{pattern}': {[f.name for f in files]}")
    
    # Remove duplicates and sort
    model_files = list(set(model_files))
    model_files.sort()
    
    if not model_files:
        logger.warning(f"No pytorch_model.bin or safetensors files found in {model_path}")
        logger.info("Available files in directory:")
        for file in model_path.iterdir():
            if file.is_file():
                logger.info(f"  - {file.name}")
    else:
        logger.info(f"Total model files found: {len(model_files)}")
        for file in model_files:
            file_size = file.stat().st_size / (1024**3)  # Size in GB
            logger.info(f"  - {file.name} ({file_size:.2f} GB)")
    
    return model_files

def validate_model_structure(model_path):
    """Validate that the model directory has required files"""
    model_path = Path(model_path)
    
    required_files = ["config.json"]
    optional_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    
    missing_required = []
    for required_file in required_files:
        if not (model_path / required_file).exists():
            missing_required.append(required_file)
    
    if missing_required:
        logger.error(f"Missing required files: {missing_required}")
        return False
    
    found_optional = []
    for optional_file in optional_files:
        if (model_path / optional_file).exists():
            found_optional.append(optional_file)
    
    if found_optional:
        logger.info(f"Found optional files: {found_optional}")
    
    return True

def get_local_model_name(model_id):
    """Convert model ID to local directory name"""
    # naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B -> HyperCLOVAX-SEED-Text-Instruct-0.5B
    if "/" in model_id:
        return model_id.split("/")[-1]
    return model_id

def main():
    parser = argparse.ArgumentParser(description="Optimize LLM model for Smart Pub")
    parser.add_argument("--model-id", default=MODEL_ID, help="Model ID to optimize")
    parser.add_argument("--model-path", help="Local path to model directory (overrides model-id)")
    parser.add_argument("--method", choices=["int8", "int4", "onnx"], required=True, help="Optimization method")
    parser.add_argument("--force", action="store_true", help="Force optimization even if model files are not found")
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = Path(args.model_path)
        model_id_for_optimizer = str(model_path)
        logger.info(f"Using local model path: {model_path}")
    else:
        # Convert model ID to local directory name
        local_model_name = get_local_model_name(args.model_id)
        model_path = Path(MODEL_DIR) / local_model_name
        model_id_for_optimizer = str(model_path)  # Use local path instead of HF model ID
        logger.info(f"Using model ID: {args.model_id}")
        logger.info(f"Local model name: {local_model_name}")
        logger.info(f"Expected model path: {model_path}")
    
    # Check if model path exists
    if not model_path.exists():
        logger.error(f"Model directory does not exist: {model_path}")
        logger.info(f"Download the model first:")
        if args.model_path:
            logger.info(f"Create directory and download model files to: {model_path}")
        else:
            logger.info(f"python3 -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{args.model_id}', local_dir='{model_path}')\"")
        if not args.force:
            sys.exit(1)
    
    # Validate model directory structure
    if not validate_model_structure(model_path):
        if not args.force:
            logger.error("Model validation failed. Use --force to proceed anyway.")
            sys.exit(1)
        else:
            logger.warning("Model validation failed, but proceeding due to --force flag")
    
    # Find model files
    model_files = find_model_files(model_path)
    
    if not model_files and not args.force:
        logger.error("No model files found. Use --force to proceed anyway.")
        logger.info("Make sure you have downloaded the model files using commands like:")
        if not args.model_path:
            logger.info(f"python3 -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{args.model_id}', local_dir='{model_path}')\"")
        sys.exit(1)
    elif not model_files:
        logger.warning("No model files found, but proceeding due to --force flag")
    
    logger.info(f"Optimizing model at {model_path} with {args.method}")
    
    try:
        # Always pass the local path to optimizer
        optimizer = ModelOptimizer(model_id_for_optimizer, MODEL_DIR)
        
        optimized_model_dir = optimizer.optimize(args.method)
        
        if optimized_model_dir:
            logger.info(f"Successfully optimized model. Saved to {optimized_model_dir}")
            
            # Verify optimized model files
            optimized_path = Path(optimized_model_dir)
            if optimized_path.exists():
                logger.info("Optimized model files:")
                for file in optimized_path.iterdir():
                    if file.is_file():
                        file_size = file.stat().st_size / (1024**2)  # Size in MB
                        logger.info(f"  - {file.name} ({file_size:.2f} MB)")
        else:
            logger.error("Failed to optimize model")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        if "not a local folder" in str(e) or "not a valid model identifier" in str(e):
            logger.info("This error suggests the optimizer is trying to download from Hugging Face.")
            logger.info("Make sure your ModelOptimizer class is configured to use local paths.")
            logger.info(f"Local model path: {model_path}")
        elif "pytorch_model.bin" in str(e) or "safetensors" in str(e):
            logger.info("This error might be related to missing model files.")
            logger.info("Try downloading the model files first:")
            if not args.model_path:
                logger.info(f"python3 -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='{args.model_id}', local_dir='{model_path}')\"")
        sys.exit(1)

if __name__ == "__main__":
    main()