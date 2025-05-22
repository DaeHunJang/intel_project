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
from smart_pub.config import DATA_DIR, VECTOR_DB_DIR

logger = get_logger("vector_db_builder")

def main():
    # Process drinks data
    logger.info("Processing drinks data...")
    drink_processor = DrinkProcessor(DATA_DIR)
    processed_drinks = drink_processor.process()
    
    if not processed_drinks:
        logger.error("Failed to process drinks data")
        return
    
    logger.info(f"Processed {len(processed_drinks)} drinks")
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore(VECTOR_DB_DIR)
    
    # Load embedding model
    logger.info("Loading embedding model...")
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