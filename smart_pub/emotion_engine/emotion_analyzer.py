from typing import List, Dict, Any
from ..model.llm_wrapper import LLMWrapper
from ..utils.helpers import get_logger
from ..config import EMOTION_ANALYSIS_TEMPLATE, EMOTION_KEYWORDS

logger = get_logger(__name__)

class EmotionAnalyzer:
    def __init__(self, llm: LLMWrapper):
        self.llm = llm
        self.emotion_keywords = EMOTION_KEYWORDS
    
    def analyze(self, user_message: str) -> List[str]:
        prompt = EMOTION_ANALYSIS_TEMPLATE.format(
            emotion_keywords=", ".join(self.emotion_keywords),
            user_message=user_message
        )
        
        response = self.llm.generate(prompt)
        if not response:
            logger.warning("Empty response from emotion analysis")
            return ["기분전환"]
        
        emotions = []
        for keyword in self.emotion_keywords:
            if keyword in response:
                emotions.append(keyword)
        
        if not emotions:
            logger.warning(f"No emotions found in the response: {response}")
            return ["기분전환"]
        
        logger.info(f"Detected emotions: {emotions}")
        return emotions[:3]