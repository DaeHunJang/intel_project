import os
from typing import Dict, List, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import warnings

# GPU 파이프라인 경고 무시
warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")

from ..utils.helpers import get_logger
from ..config import MODEL_ID, MODEL_DIR, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, MODEL_QUANTIZATION

logger = get_logger(__name__)

class LLMWrapper:
    def __init__(self, model_id: str = MODEL_ID, model_dir: str = MODEL_DIR):
        self.model_id = model_id
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self) -> bool:
        try:
            model_basename = os.path.basename(self.model_id)
            quantization_config = None
            
            if "-int8" in model_basename:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                logger.info(f"Using INT8 quantization for {self.model_id}")
            elif "-int4" in model_basename:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info(f"Using INT4 quantization for {self.model_id}")
            elif MODEL_QUANTIZATION == "int8":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                logger.info(f"Using INT8 quantization from config for {self.model_id}")
            elif MODEL_QUANTIZATION == "int4":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info(f"Using INT4 quantization from config for {self.model_id}")
            
            model_path = self.model_id
            if not os.path.exists(model_path) and not model_path.startswith("/"):
                model_path = self.model_id
                logger.info(f"Loading model from Hugging Face Hub: {model_path}")
            else:
                logger.info(f"Loading model from local path: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.model_dir,
                padding_side="left",
                truncation_side="left"
            )
            
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir=self.model_dir,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if not quantization_config else None
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Successfully loaded model {self.model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model {self.model_id}: {e}")
            return False
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """단일 프롬프트 생성"""
        if not self.pipeline:
            logger.error("Model is not loaded. Call load_model() first.")
            return ""
        
        try:
            generation_kwargs = {
                "max_new_tokens": max_new_tokens or MAX_NEW_TOKENS,
                "temperature": temperature or TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": True,
                "return_full_text": False
            }
            
            response = self.pipeline(prompt, **generation_kwargs)
            generated_text = response[0]["generated_text"]
            
            if not generated_text.strip():
                return ""
            
            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    def generate_batch(self, prompts: List[str], max_new_tokens: Optional[int] = None, temperature: Optional[float] = None, batch_size: int = 4) -> List[str]:
        """배치로 여러 프롬프트를 동시에 처리 - GPU 효율성 향상"""
        if not self.pipeline:
            logger.error("Model is not loaded. Call load_model() first.")
            return [""] * len(prompts)
        
        try:
            generation_kwargs = {
                "max_new_tokens": max_new_tokens or MAX_NEW_TOKENS,
                "temperature": temperature or TEMPERATURE,
                "top_p": TOP_P,
                "do_sample": True,
                "return_full_text": False,
                "batch_size": min(batch_size, len(prompts))
            }
            
            logger.debug(f"Processing {len(prompts)} prompts in batch (batch_size={generation_kwargs['batch_size']})")
            
            # 배치 처리
            responses = self.pipeline(prompts, **generation_kwargs)
            
            # 결과 추출
            results = []
            for i, response in enumerate(responses):
                try:
                    if isinstance(response, list):
                        generated_text = response[0]["generated_text"] if response else ""
                    else:
                        generated_text = response.get("generated_text", "")
                    
                    results.append(generated_text.strip() if generated_text else "")
                except Exception as e:
                    logger.warning(f"Error processing response {i}: {e}")
                    results.append("")
            
            logger.debug(f"Batch generation completed: {len(results)} results")
            return results
        
        except Exception as e:
            logger.error(f"Error generating batch text: {e}")
            logger.info("Falling back to individual generation...")
            
            # 폴백: 개별 처리
            results = []
            for prompt in prompts:
                try:
                    result = self.generate(prompt, max_new_tokens, temperature)
                    results.append(result)
                except Exception as e2:
                    logger.warning(f"Individual generation also failed: {e2}")
                    results.append("")
            
            return results
    
    def generate_with_retry(self, prompt: str, max_retries: int = 3, **kwargs) -> str:
        """재시도 로직이 있는 생성 메서드"""
        for attempt in range(max_retries):
            try:
                result = self.generate(prompt, **kwargs)
                if result and len(result.strip()) > 0:
                    return result
                else:
                    logger.warning(f"Empty result on attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed")
        
        return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self.model:
            return {"status": "not_loaded"}
        
        info = {
            "model_id": self.model_id,
            "status": "loaded",
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "dtype": str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
        }
        
        # GPU 메모리 정보
        if torch.cuda.is_available():
            try:
                current_device = torch.cuda.current_device()
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated(current_device) / (1024**2)
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved(current_device) / (1024**2)
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return info