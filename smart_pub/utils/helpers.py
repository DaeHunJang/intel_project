import json
import logging
from typing import Dict, List, Any, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def load_json(file_path: str) -> Union[Dict, List]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = get_logger("utils")
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return {}

def save_json(data: Union[Dict, List], file_path: str) -> bool:
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger = get_logger("utils")
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False

def extract_non_null_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}