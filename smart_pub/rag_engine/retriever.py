from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from ..model.llm_wrapper import LLMWrapper
from ..utils.helpers import get_logger
from ..config import RAG_TEMPLATE, RETRIEVAL_TOP_K

logger = get_logger(__name__)

class Retriever:
    def __init__(self, vector_store: VectorStore, llm: LLMWrapper):
        self.vector_store = vector_store
        self.llm = llm
    
    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, top_k)
    
    def retrieve_drink(self, drink_id: str) -> Optional[Dict[str, Any]]:
        for doc in self.vector_store.documents:
            if doc["drink_id"] == drink_id and doc["type"] == "basic":
                return doc["drink"]
        
        return None
    
    def generate_response(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> str:
        documents = self.retrieve(query, top_k)
        
        if not documents:
            logger.warning(f"No documents found for query: {query}")
            return "죄송합니다, 그 질문에 대한 정보를 찾을 수 없습니다."
        
        context = ""
        for i, doc in enumerate(documents):
            context += f"문서 {i+1}:\n{doc['text']}\n\n"
        
        prompt = RAG_TEMPLATE.format(
            context=context,
            question=query
        )
        
        response = self.llm.generate(prompt)
        
        return response