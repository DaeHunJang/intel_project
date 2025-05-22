import os
from typing import Dict, List, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
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
            # Determine quantization config based on model path if it contains quantization info
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
            
            # Handle model loaded from local directory or from hub
            model_path = self.model_id
            if not os.path.exists(model_path) and not model_path.startswith("/"):
                # This is a model ID, not a local path
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