import os
import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.data_processing.drink_processor import DrinkProcessor
from smart_pub.model.llm_wrapper import LLMWrapper
from smart_pub.model.model_evaluator import ModelEvaluationViz
from smart_pub.emotion_engine.emotion_analyzer import EmotionAnalyzer
from smart_pub.rag_engine.vector_store import VectorStore
from smart_pub.rag_engine.retriever import Retriever
from smart_pub.recommendation_engine.recommender import Recommender
from smart_pub.conversation.dialogue_manager import DialogueManager
from smart_pub.utils.helpers import get_logger
from smart_pub.config import DATA_DIR, MODEL_DIR, VECTOR_DB_DIR, RESULTS_DIR, EMBEDDING_MODEL_ID

logger = get_logger("main")

def check_embedding_model():
    """Check if embedding model exists locally"""
    if "/" in EMBEDDING_MODEL_ID:
        model_name = EMBEDDING_MODEL_ID.split("/")[-1]
    else:
        model_name = EMBEDDING_MODEL_ID
    
    local_path = Path(MODEL_DIR) / model_name
    
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Embedding model found: {local_path}")
        return True
    else:
        logger.warning(f"Embedding model not found at {local_path}")
        logger.info("Run the following command to download:")
        logger.info("python download_embedding_model.py")
        return False

def setup_argparse():
    parser = argparse.ArgumentParser(description="Smart Pub CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--rebuild-vectordb", action="store_true", help="Rebuild vector database")
    parser.add_argument("--model-id", help="Override model ID to use")
    parser.add_argument("--skip-embedding-check", action="store_true", help="Skip embedding model check")
    return parser.parse_args()

def initialize_app(args):
    # Check embedding model first
    if not args.skip_embedding_check:
        if not check_embedding_model():
            logger.error("Embedding model not found. Please download it first.")
            return None
    
    # Process drinks data
    logger.info("Processing drinks data...")
    drink_processor = DrinkProcessor(DATA_DIR)
    processed_drinks = drink_processor.process()
    
    if not processed_drinks:
        logger.error("Failed to process drinks data")
        return None
    
    # Determine which model to use
    model_id = args.model_id
    if not model_id:
        # Try to get best model from evaluation results
        viz = ModelEvaluationViz(RESULTS_DIR)
        best_model = viz.get_best_model()
        
        if best_model:
            logger.info(f"Using best model from evaluation: {best_model}")
            model_id = best_model
        else:
            # Fall back to default model in config
            from smart_pub.config import MODEL_ID
            model_id = MODEL_ID
            logger.info(f"Using default model from config: {model_id}")
    
    # Load LLM model
    logger.info(f"Loading LLM model: {model_id}")
    llm = LLMWrapper(model_id=model_id)
    if not llm.load_model():
        logger.error("Failed to load LLM model")
        return None
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = VectorStore()
    
    # Check if vector database exists
    vector_db_exists = os.path.exists(os.path.join(VECTOR_DB_DIR, "faiss_index.bin"))
    
    if not vector_db_exists or args.rebuild_vectordb:
        logger.info("Building vector database...")
        vector_store.load_embedding_model()
        if not vector_store.build_index(processed_drinks):
            logger.error("Failed to build vector database")
            return None
    else:
        logger.info("Loading existing vector database...")
        if not vector_store.load(processed_drinks):
            logger.error("Failed to load vector database")
            return None
    
    # Initialize components
    logger.info("Initializing components...")
    emotion_analyzer = EmotionAnalyzer(llm)
    retriever = Retriever(vector_store, llm)
    recommender = Recommender(retriever, emotion_analyzer, llm)
    dialogue_manager = DialogueManager(recommender, retriever, llm)
    
    logger.info("Initialization complete")
    return dialogue_manager

def run_cli_interface(dialogue_manager):
    print("\n" + "="*50)
    print("Smart Pub AI 바텐더에 오신 것을 환영합니다!")
    print("감정을 표현하거나 원하는 것을 말씀해주세요.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\033[94m사용자:\033[0m ")
            
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("\n\033[93m감사합니다. 필요하시면 다시 불러주세요!\033[0m")
                break
            
            if not user_input.strip():
                continue
            
            response = dialogue_manager.generate_response(user_input)
            print(f"\033[92mAI 바텐더:\033[0m {response}\n")
            
        except KeyboardInterrupt:
            print("\n\033[93m프로그램을 종료합니다. 다음에 또 만나요!\033[0m")
            break
        
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print("\033[91m죄송합니다, 오류가 발생했습니다. 다시 시도해주세요.\033[0m")

def main():
    args = setup_argparse()
    dialogue_manager = initialize_app(args)
    
    if dialogue_manager:
        run_cli_interface(dialogue_manager)
    else:
        logger.error("Failed to initialize application")
        sys.exit(1)

if __name__ == "__main__":
    main()