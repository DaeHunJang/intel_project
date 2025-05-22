import os
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

from ..utils.helpers import load_json, save_json, get_logger

logger = get_logger(__name__)

class DrinkProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.drinks_file = self.data_dir / "drinks.json"
        self.processed_file = self.data_dir / "drinks_processed.json"
        self.drinks_data = []
        
    def load_drinks(self) -> bool:
        if os.path.exists(self.drinks_file):
            data = load_json(self.drinks_file)
            if isinstance(data, list):
                self.drinks_data = data
                logger.info(f"Loaded {len(self.drinks_data)} drinks from {self.drinks_file}")
                return True
            else:
                logger.error(f"Invalid format in {self.drinks_file}. Expected a list.")
        else:
            logger.error(f"Drinks file not found: {self.drinks_file}")
        return False
    
    def preprocess_drinks(self) -> List[Dict[str, Any]]:
        processed_drinks = []
        
        for drink in self.drinks_data:
            processed_drink = {
                "id": drink.get("idDrink"),
                "name": drink.get("strDrink"),
                "category": drink.get("strCategory"),
                "alcoholic": drink.get("strAlcoholic"),
                "glass": drink.get("strGlass"),
                "instructions": drink.get("strInstructions"),
                "image_url": drink.get("strDrinkThumb"),
                "tags": drink.get("strTags", "").split(",") if drink.get("strTags") else []
            }
            
            ingredients = []
            for i in range(1, 16):
                ingredient_key = f"strIngredient{i}"
                measure_key = f"strMeasure{i}"
                
                ingredient = drink.get(ingredient_key)
                measure = drink.get(measure_key, "").strip() if drink.get(measure_key) else ""
                
                if ingredient:
                    ingredients.append({
                        "ingredient": ingredient,
                        "measure": measure
                    })
            
            processed_drink["ingredients"] = ingredients
            processed_drink["tags"] = [tag.strip() for tag in processed_drink["tags"] if tag.strip()]
            processed_drinks.append(processed_drink)
        
        logger.info(f"Preprocessed {len(processed_drinks)} drinks")
        return processed_drinks
    
    def save_processed_drinks(self, processed_drinks: List[Dict[str, Any]]) -> bool:
        return save_json(processed_drinks, self.processed_file)
    
    def get_drink_dataframe(self) -> pd.DataFrame:
        if os.path.exists(self.processed_file):
            processed_drinks = load_json(self.processed_file)
            return pd.DataFrame(processed_drinks)
        else:
            logger.warning(f"Processed drinks file not found: {self.processed_file}")
            return pd.DataFrame()
    
    def process(self) -> List[Dict[str, Any]]:
        if self.load_drinks():
            processed_drinks = self.preprocess_drinks()
            if self.save_processed_drinks(processed_drinks):
                logger.info(f"Saved processed drinks to {self.processed_file}")
            return processed_drinks
        return []