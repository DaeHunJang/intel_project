from typing import List, Dict, Any, Optional, Tuple
from ..recommendation_engine.recommender import Recommender
from ..rag_engine.retriever import Retriever
from ..model.llm_wrapper import LLMWrapper
from ..utils.helpers import get_logger
from ..config import CONVERSATION_TEMPLATE

logger = get_logger(__name__)

class DialogueManager:
    def __init__(self, recommender: Recommender, retriever: Retriever, llm: LLMWrapper):
        self.recommender = recommender
        self.retriever = retriever
        self.llm = llm
        self.conversation_history = []
        self.current_recommendation = None
        self.context = {}
    
    def add_message(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def format_conversation_history(self) -> str:
        formatted = ""
        for message in self.conversation_history:
            if message["role"] == "user":
                formatted += f"사용자: {message['content']}\n"
            else:
                formatted += f"AI 바텐더: {message['content']}\n"
        return formatted
    
    def generate_response(self, user_message: str) -> str:
        self.add_message("user", user_message)
        
        # Check if response requires RAG-based knowledge
        if "재료" in user_message or "만드는 법" in user_message or "어떻게 만들" in user_message:
            response = self.retriever.generate_response(user_message)
        
        # Check if user is asking for a recommendation
        elif "추천" in user_message or "어떤 술" in user_message or "뭐 마실까" in user_message:
            recommendation = self.recommender.recommend(user_message, 
                                                      self.current_recommendation["drink"]["id"] if self.current_recommendation else None)
            
            if recommendation["success"]:
                self.current_recommendation = recommendation
                response = recommendation["explanation"]
            else:
                response = "죄송합니다, 감정에 맞는 적절한 음료를 찾지 못했습니다. 다른 감정을 표현해주시겠어요?"
        
        # Check if user is rejecting current recommendation
        elif self.current_recommendation and ("다른" in user_message or "싫어" in user_message or "별로" in user_message):
            recommendation = self.recommender.recommend(user_message, 
                                                      self.current_recommendation["drink"]["id"] if self.current_recommendation else None)
            
            if recommendation["success"]:
                self.current_recommendation = recommendation
                response = (
                    f"다른 추천을 원하시는군요. {recommendation['drink']['name']}는 어떨까요? "
                    f"{recommendation['explanation']}"
                )
            else:
                response = "죄송합니다, 다른 적절한 음료를 찾지 못했습니다. 다시 말씀해주시겠어요?"
        
        # Check if user is accepting current recommendation
        elif self.current_recommendation and ("좋아" in user_message or "주문" in user_message or "시킬게" in user_message):
            response = (
                f"{self.current_recommendation['drink']['name']} 주문이 완료되었습니다! "
                f"곧 준비해서 가져다 드리겠습니다. 맛있게 즐기세요!"
            )
        
        # General conversation
        else:
            prompt = CONVERSATION_TEMPLATE.format(
                conversation_history=self.format_conversation_history(),
                user_message=user_message
            )
            
            response = self.llm.generate(prompt)

            # 여기서 "사용자:"가 포함되어 있으면 그 앞까지만 자르기
            if "사용자:" in response:
                response = response.split("사용자:")[0].strip()
            if "User:" in response:  # 혹시 영어로 나올 경우도 대비
                response = response.split("User:")[0].strip()

            if not response:
                # fallback 로직
                recommendation = self.recommender.recommend(user_message)
                if recommendation["success"]:
                    self.current_recommendation = recommendation
                    response = (
                        f"저는 당신의 감정을 분석해서 적절한 음료를 추천해드리고 있어요. "
                        f"{recommendation['explanation']}"
                    )
                else:
                    response = "죄송합니다, 이해하지 못했습니다. 감정을 표현하거나 술 추천을 요청해주시겠어요?"
        
        self.add_message("assistant", response)
        return response
    
    def clear_history(self) -> None:
        self.conversation_history = []
        self.current_recommendation = None
        self.context = {}