#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, login, HfApi
import shutil
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.utils.helpers import get_logger
from smart_pub.config import MODEL_ID, MODEL_DIR, MODEL_CANDIDATES

logger = get_logger("model_downloader")

def download_model(model_id, model_dir=MODEL_DIR, token=None):
    """
    Download the model from Hugging Face Hub
    """
    try:
        target_dir = os.path.join(model_dir, os.path.basename(model_id))
        logger.info(f"Downloading model {model_id} to {target_dir}")
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        
        # First check if we have access to the model
        try:
            # Use HfApi to check if model exists and we have access
            api = HfApi(token=token)
            model_info = api.model_info(model_id)
            if model_info.private and not token:
                logger.error(f"Model {model_id} is private and requires a token")
                print(f"\n⚠️ 모델 {model_id}는 비공개 모델이며 접근 토큰이 필요합니다.")
                return False
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "unauthorized" in str(e).lower():
                logger.error(f"Authorization error for model {model_id}. You may need to request access to this model.")
                print(f"\n⚠️ 접근 권한 문제: {model_id}")
                print("허깅페이스 웹사이트에서 이 모델에 대한 접근 권한을 요청해야 할 수 있습니다.")
                print(f"다음 URL을 방문하세요: https://huggingface.co/{model_id}")
                print("모델 페이지에서 'Access Request' 버튼을 찾아 접근 요청을 제출하세요.")
                print("접근 승인을 받은 후 다시 시도하세요.\n")
                return False
            elif "404" in str(e):
                logger.error(f"Model {model_id} not found")
                print(f"\n⚠️ 모델을 찾을 수 없음: {model_id}")
                return False
            else:
                logger.error(f"Error checking model {model_id}: {e}")
                print(f"\n⚠️ 모델 확인 중 오류 발생: {model_id}")
                return False
        
        # Download the model - disable progress bars to avoid tqdm issues
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            ignore_patterns=["*.pt", "*.bin", "*.h5", "*.ot", "*.safetensors"],
            token=token,
            local_files_only=False,
            tqdm_class=None  # Disable tqdm progress bar
        )
        
        logger.info(f"Successfully downloaded model {model_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {e}")
        return False

def download_all_models(models=None, token=None):
    """
    Download all models sequentially (to avoid tqdm issues)
    """
    if models is None:
        models = MODEL_CANDIDATES
    
    success_count = 0
    failed_models = []
    
    logger.info(f"Downloading {len(models)} models sequentially...")
    
    # Sequential download instead of parallel to avoid tqdm issues
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 다운로드 중: {model}")
        success = download_model(model, MODEL_DIR, token)
        
        if success:
            success_count += 1
            print(f"{model} 다운로드 성공")
        else:
            failed_models.append(model)
            print(f"{model} 다운로드 실패")
    
    logger.info(f"Downloaded {success_count} of {len(models)} models successfully")
    
    if failed_models:
        logger.warning(f"Failed to download the following models: {failed_models}")
    
    return success_count, failed_models

def main():
    parser = argparse.ArgumentParser(description="Download LLM models for Smart Pub")
    parser.add_argument("--model-id", help="Specific model ID to download")
    parser.add_argument("--all", action="store_true", help="Download all candidate models")
    parser.add_argument("--list-models", action="store_true", help="List all available models")
    parser.add_argument("--token", help="Hugging Face API token")
    args = parser.parse_args()
    
    # Get token from args, env var, or ask user
    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        token_input = input("Hugging Face API 토큰이 필요합니다. 토큰을 입력해주세요 (입력하지 않으면 로그인 절차를 진행합니다): ")
        if token_input.strip():
            token = token_input
        else:
            # Use huggingface_hub login functionality
            print("허깅페이스 로그인 절차를 시작합니다...")
            login()
    else:
        # If token is provided, explicitly log in with it
        login(token=token)
        print("허깅페이스 토큰으로 로그인되었습니다.")
    
    if args.list_models:
        print("\n사용 가능한 모델 목록:")
        for i, model in enumerate(MODEL_CANDIDATES, 1):
            print(f"{i}. {model}")
        return
    
    if args.all:
        # Download all models
        success_count, failed_models = download_all_models(token=token)
        
        if success_count == len(MODEL_CANDIDATES):
            print("\n모든 모델이 성공적으로 다운로드되었습니다!")
        else:
            print(f"\n{len(MODEL_CANDIDATES)}개 중 {success_count}개 모델이 다운로드되었습니다.")
            if failed_models:
                print("\n다운로드 실패한 모델:")
                for model in failed_models:
                    print(f"  - {model}")
                print("\n각 모델에 대한 접근 권한을 확인하세요: https://huggingface.co/")
    else:
        # Download single model
        model_id = args.model_id or MODEL_ID
        if download_model(model_id, token=token):
            print(f"\n모델 {model_id}가 성공적으로 다운로드되었습니다!")
        else:
            print(f"\n모델 {model_id} 다운로드에 실패했습니다.")
            sys.exit(1)

if __name__ == "__main__":
    main()