import os
import argparse
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smart_pub.data_processing.drink_processor import DrinkProcessor
from smart_pub.model.llm_wrapper import LLMWrapper
from smart_pub.emotion_engine.emotion_analyzer import EmotionAnalyzer
from smart_pub.rag_engine.vector_store import VectorStore
from smart_pub.rag_engine.retriever import Retriever
from smart_pub.recommendation_engine.recommender import Recommender
from smart_pub.conversation.dialogue_manager import DialogueManager
from smart_pub.utils.helpers import get_logger
from smart_pub.config import DATA_DIR, MODEL_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL_ID, MODEL_ID

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
        logger.info("python download_operational_model.py --embedding")
        return False

def get_local_model_path(model_id: str, model_dir: str) -> str | None:
    """Convert model ID to local path"""
    if "/" in model_id:
        model_name = model_id.split("/")[-1]
    else:
        model_name = model_id
    
    local_path = Path(model_dir) / model_name
    
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"Found local model: {local_path}")
        return str(local_path)
    else:
        logger.warning(f"Local model not found at {local_path}")
        return None

def list_available_models(model_dir: str) -> list:
    """List available local models"""
    model_dir_path = Path(model_dir)
    available_models = []
    
    if not model_dir_path.exists():
        return available_models
    
    for item in model_dir_path.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            available_models.append(item.name)
    
    return available_models
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
        logger.info("python download_operational_model.py --embedding")
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
    
    # Determine which model to use - ONLY use local models
    requested_model = args.model_id if args.model_id else MODEL_ID
    
    # Convert to local path
    local_model_path = get_local_model_path(requested_model, MODEL_DIR)
    
    if not local_model_path:
        print(f"모델을 찾을 수 없습니다: {requested_model}")
        
        # Show available models
        available_models = list_available_models(MODEL_DIR)
        if available_models:
            print("사용 가능한 로컬 모델:")
            for i, model in enumerate(available_models, 1):
                print(f"  {i}. {model}")
            print(f"\nconfig.py에서 MODEL_ID를 다음 중 하나로 변경하세요:")
            for model in available_models:
                print(f"  MODEL_ID = \"{model}\"")
        else:
            print("로컬 모델이 없습니다.")
            print("다음 명령으로 모델을 다운로드하세요:")
            print("  python download_operational_model.py")
        
        return None
    
    print(f"로컬 모델 사용: {local_model_path}")
    
    # Load LLM model with local path
    logger.info(f"Loading LLM model from: {local_model_path}")
    llm = LLMWrapper(model_id=local_model_path, model_dir=MODEL_DIR)
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
    print("Smart Pub AI 바텐더")
    print("감정을 표현하거나 원하는 것을 말씀해주세요.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("사용자: ")
            
            if user_input.lower() in ["exit", "quit", "종료", "나가기"]:
                print("\n감사합니다. 안녕히 가세요!")
                break
            
            if not user_input.strip():
                print("메시지를 입력해주세요.")
                continue
            
            response = dialogue_manager.generate_response(user_input)
            print(f"AI 바텐더: {response}\n")
            
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            print("죄송합니다, 오류가 발생했습니다. 다시 시도해주세요.")

def main():
    print("Smart Pub 앱을 시작합니다...")
    
    args = setup_argparse()
    
    if args.debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
        print("디버그 모드 활성화")
    
    dialogue_manager = initialize_app(args)
    
    if dialogue_manager:
        # Show model info
        model_info = dialogue_manager.recommender.llm.get_model_info()
        if model_info.get("status") == "loaded":
            print("모델 로드 완료")
            print(f"  모델: {model_info.get('model_id', 'Unknown')}")
            print(f"  디바이스: {model_info.get('device', 'Unknown')}")
            if 'gpu_memory_allocated' in model_info:
                print(f"  GPU 메모리: {model_info['gpu_memory_allocated']:.1f}MB")
        
        run_cli_interface(dialogue_manager)
    else:
        logger.error("Failed to initialize application")
        print("앱 초기화에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()