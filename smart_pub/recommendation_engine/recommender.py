from typing import List, Dict, Any, Optional
import random
from ..rag_engine.retriever import Retriever
from ..emotion_engine.emotion_analyzer import EmotionAnalyzer
from ..model.llm_wrapper import LLMWrapper
from ..utils.helpers import get_logger
from ..config import EMOTION_DRINK_MAPPING, DRINK_RECOMMENDATION_TEMPLATE

logger = get_logger(__name__)

class Recommender:
    def __init__(self, retriever: Retriever, emotion_analyzer: EmotionAnalyzer, llm: LLMWrapper):
        self.retriever = retriever
        self.emotion_analyzer = emotion_analyzer
        self.llm = llm
        self.recommended_drinks = set()
    
    def analyze_emotion(self, user_message: str) -> List[str]:
        return self.emotion_analyzer.analyze(user_message)
    
    def get_drinks_by_emotion(self, emotions: List[str]) -> List[Dict[str, Any]]:
        if not emotions:
            return []
        
        drink_candidates = []
        
        for emotion in emotions:
            if emotion in EMOTION_DRINK_MAPPING:
                tags = EMOTION_DRINK_MAPPING[emotion]
                
                for doc in self.retriever.vector_store.documents:
                    if doc["type"] == "basic":
                        drink = doc["drink"]
                        drink_tags = [tag.strip() for tag in drink.get("tags", [])]
                        
                        # Check if drink tags match any emotion tags
                        match = False
                        for tag in tags:
                            if any(tag.lower() in drink_tag.lower() for drink_tag in drink_tags):
                                match = True
                                break
                        
                        if match and drink["id"] not in self.recommended_drinks:
                            drink_candidates.append(drink)
        
        # If no matching drinks or all have been recommended, reset history
        if not drink_candidates:
            self.recommended_drinks = set()
            
            # Try again but ignore recommendation history
            for emotion in emotions:
                if emotion in EMOTION_DRINK_MAPPING:
                    tags = EMOTION_DRINK_MAPPING[emotion]
                    
                    for doc in self.retriever.vector_store.documents:
                        if doc["type"] == "basic":
                            drink = doc["drink"]
                            drink_tags = [tag.strip() for tag in drink.get("tags", [])]
                            
                            match = False
                            for tag in tags:
                                if any(tag.lower() in drink_tag.lower() for drink_tag in drink_tags):
                                    match = True
                                    break
                            
                            if match:
                                drink_candidates.append(drink)
        
        return drink_candidates
    
    def recommend(self, user_message: str, previous_recommendation: Optional[str] = None) -> Dict[str, Any]:
        emotions = self.analyze_emotion(user_message)
        logger.info(f"Detected emotions: {emotions}")
        
        drink_candidates = self.get_drinks_by_emotion(emotions)
        logger.info(f"Found {len(drink_candidates)} drink candidates")
        
        if not drink_candidates:
            return {
                "success": False,
                "emotions": emotions,
                "drink": None,
                "explanation": "No suitable drinks found for the current emotions."
            }
        
        # Filter out previous recommendation if provided
        if previous_recommendation:
            drink_candidates = [drink for drink in drink_candidates if drink["id"] != previous_recommendation]
            
            if not drink_candidates:
                drink_candidates = self.get_drinks_by_emotion(emotions)
        
        # Select a random drink from candidates
        selected_drink = random.choice(drink_candidates)
        
        # Add to recommendation history
        self.recommended_drinks.add(selected_drink["id"])
        
        # Generate explanation using LLM
        prompt = DRINK_RECOMMENDATION_TEMPLATE.format(
            emotion=", ".join(emotions)
        )
        
        explanation = self.llm.generate(prompt)
        
        if not explanation or len(explanation) < 10:
            explanation = (
                f"{selected_drink['name']}는 {', '.join(emotions)} 감정을 가진 분에게 적합한 음료입니다. "
                f"이 음료는 {selected_drink['alcoholic']} 타입이며, {selected_drink['glass']}에 제공됩니다."
            )
        
        return {
            "success": True,
            "emotions": emotions,
            "drink": selected_drink,
            "explanation": explanation
        }