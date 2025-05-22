import os
from typing import Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from optimum.onnxruntime import ORTModelForCausalLM

from ..utils.helpers import get_logger

logger = get_logger(__name__)

class ModelOptimizer:
    def __init__(self, model_id: str, model_dir: str):
        self.model_id = model_id
        self.model_dir = model_dir
        
    def quantize_int8(self) -> str:
        output_dir = os.path.join(self.model_dir, f"{os.path.basename(self.model_id)}-int8")
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quantization_config
            )
            
            model.save_pretrained(output_dir)
            logger.info(f"Quantized model to INT8 and saved to {output_dir}")
            
            return output_dir
        
        except Exception as e:
            logger.error(f"Error quantizing model to INT8: {e}")
            return ""
    
    def quantize_int4(self) -> str:
        output_dir = os.path.join(self.model_dir, f"{os.path.basename(self.model_id)}-int4")
        
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quantization_config
            )
            
            model.save_pretrained(output_dir)
            logger.info(f"Quantized model to INT4 and saved to {output_dir}")
            
            return output_dir
        
        except Exception as e:
            logger.error(f"Error quantizing model to INT4: {e}")
            return ""
    
    def convert_to_onnx(self) -> str:
        output_dir = os.path.join(self.model_dir, f"{os.path.basename(self.model_id)}-onnx")
        
        try:
            # 최신 버전의 optimum 라이브러리 사용
            # ORTModelForCausalLM로 직접 변환
            model = ORTModelForCausalLM.from_pretrained(
                self.model_id,
                export=True,
                provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
            )
            
            model.save_pretrained(output_dir)
            logger.info(f"Converted model to ONNX and saved to {output_dir}")
            
            return output_dir
        
        except Exception as e:
            logger.error(f"Error converting model to ONNX: {e}")
            logger.error(f"ONNX 변환 실패: {str(e)}. 더 간단한 최적화 방법을 사용해보세요.")
            return ""
    
    def optimize(self, method: str = "int8") -> str:
        if method == "int8":
            return self.quantize_int8()
        elif method == "int4":
            return self.quantize_int4()
        elif method == "onnx":
            return self.convert_to_onnx()
        else:
            logger.error(f"Unsupported optimization method: {method}")
            return ""