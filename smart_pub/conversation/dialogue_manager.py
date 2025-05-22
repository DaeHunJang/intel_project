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
        self.conversation_state = "greeting"  # greeting, emotion_detected, recommendation_made, details_provided
        
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
    
    def analyze_user_intent(self, user_message: str) -> str:
        """사용자 의도 분석"""
        user_message_lower = user_message.lower()
        
        # 종료 의도
        if any(word in user_message_lower for word in ["안녕", "고마워", "감사", "잘가", "그만"]):
            return "goodbye"
        
        # 추천 요청
        if any(word in user_message for word in ["추천", "뭐 마실까", "어떤 술", "음료 추천", "뭐가 좋을까"]):
            return "request_recommendation"
        
        # 재료/만드는 법 질문
        if any(word in user_message for word in ["재료", "만드는 법", "어떻게 만들", "들어가는", "레시피"]):
            return "ask_recipe"
        
        # 주문
        if any(word in user_message for word in ["주문", "시킬게", "그걸로", "좋아", "마실게"]):
            return "order"
        
        # 다른 추천 요청
        if any(word in user_message for word in ["다른", "다른 거", "별로", "싫어", "다른 추천"]):
            return "request_alternative"
        
        # 감정 표현
        emotion_keywords = ["기분", "느낌", "상태", "마음", "스트레스", "행복", "슬프", "화나", "피곤", "신나", "우울", "좋아", "안좋아"]
        if any(keyword in user_message for keyword in emotion_keywords):
            return "express_emotion"
        
        # 일반 대화
        return "general_chat"
    
    def generate_response(self, user_message: str) -> str:
        self.add_message("user", user_message)
        
        # 사용자 의도 분석
        intent = self.analyze_user_intent(user_message)
        
        response = ""
        
        if intent == "goodbye":
            response = "감사합니다! 좋은 하루 되세요. 다음에 또 찾아주세요!"
            self.conversation_state = "ended"
            
        elif intent == "express_emotion":
            # 1단계: 감정 분석만 수행
            emotions = self.recommender.analyze_emotion(user_message)
            
            if emotions:
                emotion_str = ", ".join(emotions)
                response = f"아, {emotion_str} 감정이시군요. 이런 기분일 때 어울리는 음료를 추천해드릴까요? '추천해줘'라고 말씀해주세요."
                self.context["detected_emotions"] = emotions
                self.conversation_state = "emotion_detected"
            else:
                response = "기분을 좀 더 구체적으로 말씀해주시겠어요? 예를 들어 '기분이 좋아', '스트레스 받아' 같이요."
                
        elif intent == "request_recommendation" and self.conversation_state == "emotion_detected":
            # 2단계: 감정을 바탕으로 술 추천
            if "detected_emotions" in self.context:
                emotions = self.context["detected_emotions"]
                recommendation = self.recommender.get_drinks_by_emotion(emotions)
                
                if recommendation:
                    selected_drink = recommendation[0]  # 첫 번째 추천
                    self.current_recommendation = {
                        "drink": selected_drink,
                        "emotions": emotions
                    }
                    
                    response = f"{', '.join(emotions)} 기분에는 '{selected_drink['name']}'를 추천드립니다. 더 자세한 정보가 궁금하시면 '자세히 알려줘'라고 말씀해주세요."
                    self.conversation_state = "recommendation_made"
                else:
                    response = "죄송합니다. 적절한 음료를 찾지 못했습니다. 다른 감정을 표현해주시겠어요?"
            else:
                response = "먼저 현재 기분이나 감정을 말씀해주세요."
                
        elif intent == "request_recommendation" and not self.context.get("detected_emotions"):
            response = "어떤 음료를 추천해드릴까요? 먼저 현재 기분이나 감정을 말씀해주시면 더 적합한 추천을 해드릴 수 있어요."
            
        elif intent == "ask_recipe" and self.current_recommendation:
            # 3단계: 레시피/재료 정보 제공
            drink = self.current_recommendation["drink"]
            
            # RAG를 사용해 상세 정보 검색
            detailed_info = self.retriever.generate_response(f"{drink['name']} 재료 만드는 법")
            
            if detailed_info:
                response = detailed_info
            else:
                # 기본 정보 제공
                ingredients_text = f"{drink['name']} 재료:\n"
                for ingredient in drink.get('ingredients', []):
                    measure = ingredient.get('measure', '').strip()
                    ingredient_name = ingredient.get('ingredient', '')
                    if measure:
                        ingredients_text += f"- {ingredient_name}: {measure}\n"
                    else:
                        ingredients_text += f"- {ingredient_name}\n"
                
                instructions = drink.get('instructions', '제조법 정보가 없습니다.')
                response = f"{ingredients_text}\n만드는 법:\n{instructions}"
            
            self.conversation_state = "details_provided"
            
        elif intent == "order" and self.current_recommendation:
            drink_name = self.current_recommendation["drink"]["name"]
            response = f"{drink_name} 주문 완료! 맛있게 드세요."
            self.conversation_state = "order_completed"
            
        elif intent == "request_alternative" and self.current_recommendation:
            # 다른 추천 제공
            if "detected_emotions" in self.context:
                emotions = self.context["detected_emotions"]
                recommendations = self.recommender.get_drinks_by_emotion(emotions)
                
                # 현재 추천 제외하고 다른 것 찾기
                current_drink_id = self.current_recommendation["drink"]["id"]
                alternatives = [drink for drink in recommendations if drink["id"] != current_drink_id]
                
                if alternatives:
                    new_drink = alternatives[0]
                    self.current_recommendation["drink"] = new_drink
                    response = f"다른 추천으로 '{new_drink['name']}'는 어떠세요? 자세한 정보가 필요하시면 말씀해주세요."
                    self.conversation_state = "recommendation_made"
                else:
                    response = "죄송합니다. 다른 적절한 추천을 찾지 못했습니다."
            else:
                response = "먼저 기분을 말씀해주시면 다른 추천을 해드릴게요."
                
        else:
            # 일반 대화 또는 상황에 맞지 않는 요청
            if self.conversation_state == "greeting":
                response = "안녕하세요! Smart Pub에 오신 걸 환영합니다. 현재 기분이나 감정을 말씀해주시면 어울리는 음료를 추천해드릴게요."
            elif self.conversation_state == "emotion_detected":
                response = "감정을 파악했습니다. '추천해줘'라고 말씀하시면 어울리는 음료를 추천해드릴게요."
            elif self.conversation_state == "recommendation_made":
                response = "음료를 추천해드렸습니다. '자세히 알려줘'로 상세 정보를 보거나, '주문할게'로 주문하실 수 있어요."
            else:
                # 일반적인 대화 처리
                prompt = CONVERSATION_TEMPLATE.format(
                    conversation_history=self.format_conversation_history(),
                    user_message=user_message
                )
                response = self.llm.generate(prompt)
                
                if not response:
                    response = "죄송합니다. 다시 말씀해주시겠어요?"
        
        self.add_message("assistant", response)
        return response
    
    def clear_history(self) -> None:
        self.conversation_history = []
        self.current_recommendation = None
        self.context = {}
        self.conversation_state = "greeting"