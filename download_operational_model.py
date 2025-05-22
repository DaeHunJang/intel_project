#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.utils.helpers import get_logger
from smart_pub.config import MODEL_ID, MODEL_DIR, MODEL_CANDIDATES, EMBEDDING_MODEL_ID

logger = get_logger("model_downloader")

def download_model(model_id, model_dir=MODEL_DIR, include_weights=True):
    """Download model from Hugging Face Hub"""
    try:
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        target_dir = os.path.join(model_dir, model_name)
        
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Downloading {model_id} to {target_dir}")
        
        # Set ignore patterns based on whether we want weights
        ignore_patterns = [] if include_weights else ["*.bin", "*.safetensors", "*.h5", "*.pt", "*.ot"]
        
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            ignore_patterns=ignore_patterns
        )
        
        logger.info(f"Successfully downloaded {model_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_id}: {e}")
        return False

def download_embedding_model():
    """Download embedding model using sentence-transformers"""
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = EMBEDDING_MODEL_ID.split("/")[-1] if "/" in EMBEDDING_MODEL_ID else EMBEDDING_MODEL_ID
        target_dir = Path(MODEL_DIR) / model_name
        
        logger.info(f"Downloading embedding model: {EMBEDDING_MODEL_ID}")
        
        model = SentenceTransformer(EMBEDDING_MODEL_ID)
        model.save(str(target_dir))
        
        logger.info(f"Successfully downloaded embedding model to {target_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download embedding model: {e}")
        return False

def check_model_exists(model_id, model_dir=MODEL_DIR):
    """Check if model already exists locally"""
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    local_path = Path(model_dir) / model_name
    
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Model {model_id} already exists at {local_path}")
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Download models for Smart Pub")
    parser.add_argument("--model-id", help="Specific model ID to download")
    parser.add_argument("--all", action="store_true", help="Download all candidate models")
    parser.add_argument("--embedding", action="store_true", help="Download embedding model")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--force", action="store_true", help="Force download even if exists")
    parser.add_argument("--no-weights", action="store_true", help="Download without model weights")
    args = parser.parse_args()
    
    if args.list_models:
        print("\nLLM 모델 후보:")
        for i, model in enumerate(MODEL_CANDIDATES, 1):
            exists = "[EXISTS]" if check_model_exists(model) else "[MISSING]"
            print(f"  {i}. {model} {exists}")
        
        print(f"\n임베딩 모델:")
        exists = "[EXISTS]" if check_model_exists(EMBEDDING_MODEL_ID) else "[MISSING]"
        print(f"  - {EMBEDDING_MODEL_ID} {exists}")
        return
    
    # Login to Hugging Face if needed
    try:
        login()
    except:
        logger.warning("Hugging Face login failed, proceeding without authentication")
    
    success_count = 0
    total_count = 0
    
    if args.embedding:
        print("임베딩 모델 다운로드 중...")
        total_count += 1
        if args.force or not check_model_exists(EMBEDDING_MODEL_ID):
            if download_embedding_model():
                success_count += 1
                print(f"SUCCESS: {EMBEDDING_MODEL_ID} 다운로드 완료")
            else:
                print(f"FAILED: {EMBEDDING_MODEL_ID} 다운로드 실패")
        else:
            success_count += 1
            print(f"SKIP: {EMBEDDING_MODEL_ID} 이미 존재함")
    
    elif args.all:
        print(f"{len(MODEL_CANDIDATES)}개 모델 다운로드 중...")
        
        for i, model_id in enumerate(MODEL_CANDIDATES, 1):
            total_count += 1
            print(f"\n[{i}/{len(MODEL_CANDIDATES)}] {model_id}")
            
            if not args.force and check_model_exists(model_id):
                success_count += 1
                print(f"SKIP: 이미 존재함")
                continue
            
            if download_model(model_id, MODEL_DIR, not args.no_weights):
                success_count += 1
                print(f"SUCCESS: 다운로드 완료")
            else:
                print(f"FAILED: 다운로드 실패")
    
    elif args.model_id:
        model_id = args.model_id
        total_count += 1
        print(f"{model_id} 다운로드 중...")
        
        if not args.force and check_model_exists(model_id):
            print(f"SKIP: {model_id} 이미 존재함")
            success_count += 1
        elif download_model(model_id, MODEL_DIR, not args.no_weights):
            success_count += 1
            print(f"SUCCESS: {model_id} 다운로드 완료")
        else:
            print(f"FAILED: {model_id} 다운로드 실패")
    
    else:
        # Download default model
        model_id = MODEL_ID
        total_count += 1
        print(f"기본 모델 {model_id} 다운로드 중...")
        
        if not args.force and check_model_exists(model_id):
            print(f"SKIP: {model_id} 이미 존재함")
            success_count += 1
        elif download_model(model_id, MODEL_DIR, not args.no_weights):
            success_count += 1
            print(f"SUCCESS: {model_id} 다운로드 완료")
        else:
            print(f"FAILED: {model_id} 다운로드 실패")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"다운로드 완료: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("모든 모델이 성공적으로 준비되었습니다!")
    elif success_count > 0:
        print(f"일부 모델만 다운로드되었습니다.")
    else:
        print("모델 다운로드에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()