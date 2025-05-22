#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import json
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.data_processing.drink_processor import DrinkProcessor
from smart_pub.rag_engine.vector_store import VectorStore
from smart_pub.utils.helpers import get_logger
from smart_pub.config import DATA_DIR, VECTOR_DB_DIR, MODEL_DIR, EMBEDDING_MODEL_ID

logger = get_logger("vector_db_builder")

def get_local_embedding_model_path(model_id, model_dir):
    """Convert Hugging Face model ID to local path"""
    if "/" in model_id:
        model_name = model_id.split("/")[-1]
    else:
        model_name = model_id
    
    local_path = Path(model_dir) / model_name
    
    # Check if local model exists
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Found local embedding model: {local_path}")
        return str(local_path)
    else:
        logger.warning(f"Local embedding model not found at {local_path}")
        logger.info(f"Will use original model ID for download: {model_id}")
        return model_id

def main():
    # Process drinks data
    logger.info("Processing drinks data...")
    drink_processor = DrinkProcessor(DATA_DIR)
    processed_drinks = drink_processor.process()
    
    if not processed_drinks:
        logger.error("Failed to process drinks data")
        return
    
    logger.info(f"Processed {len(processed_drinks)} drinks")
    
    # Get local embedding model path
    embedding_model_path = get_local_embedding_model_path(EMBEDDING_MODEL_ID, MODEL_DIR)
    
    # Initialize vector store with local model path
    logger.info("Initializing vector store...")
    vector_store = VectorStore(VECTOR_DB_DIR, embedding_model_path)
    
    # Load embedding model
    logger.info(f"Loading embedding model from: {embedding_model_path}")
    if not vector_store.load_embedding_model():
        logger.error("Failed to load embedding model")
        return
    
    # Build index
    logger.info("Building vector database...")
    if not vector_store.build_index(processed_drinks):
        logger.error("Failed to build vector database")
        return
    
    logger.info(f"Successfully built vector database at {VECTOR_DB_DIR}")

if __name__ == "__main__":
    main()