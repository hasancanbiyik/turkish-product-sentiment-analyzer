"""
Translation utilities for Turkish to English
"""
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
from typing import List, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishEnglishTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-tr-en"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.translator = None
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the translation model"""
        try:
            logger.info(f"Loading translation model: {self.model_name}")
            
            # Load tokenizer and model separately for better control
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline
            self.translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            logger.info("Translation model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading translation model: {e}")
            raise
    
    def translate(self, text: Union[str, List[str]], 
                 max_length: int = 512,
                 batch_size: int = 32) -> Union[str, List[str]]:
        """
        Translate Turkish text to English
        
        Args:
            text: Single text or list of texts
            max_length: Maximum length of translation
            batch_size: Batch size for translation
            
        Returns:
            Translated text(s)
        """
        if self.translator is None:
            self.load_model()
            
        single_text = isinstance(text, str)
        if single_text:
            text = [text]
        
        # Filter empty texts
        non_empty_indices = [(i, t) for i, t in enumerate(text) if t.strip()]
        
        if not non_empty_indices:
            return "" if single_text else []
        
        texts_to_translate = [t for _, t in non_empty_indices]
        
        try:
            # Translate in batches
            translations = []
            for i in range(0, len(texts_to_translate), batch_size):
                batch = texts_to_translate[i:i + batch_size]
                
                # Translate batch
                batch_translations = self.translator(
                    batch,
                    max_length=max_length,
                    truncation=True
                )
                
                # Extract translation text
                for trans in batch_translations:
                    translations.append(trans['translation_text'])
            
            # Reconstruct full list with empty strings for filtered texts
            full_translations = [""] * len(text)
            for (original_idx, _), translation in zip(non_empty_indices, translations):
                full_translations[original_idx] = translation
            
            return full_translations[0] if single_text else full_translations
            
        except Exception as e:
            logger.error(f"Error during translation: {e}")
            # Return original text if translation fails
            return text[0] if single_text else text
    
    def translate_with_alignment(self, text: str) -> dict:
        """
        Translate and provide alignment information
        
        Returns:
            Dictionary with original, translation, and metadata
        """
        if self.translator is None:
            self.load_model()
            
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=512)
            
            # Decode translation
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'original': text,
                'translation': translation,
                'original_length': len(text.split()),
                'translation_length': len(translation.split()),
                'model_used': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error in aligned translation: {e}")
            return {
                'original': text,
                'translation': text,  # Return original if failed
                'error': str(e)
            }
    
    def batch_translate_reviews(self, reviews: List[dict]) -> List[dict]:
        """
        Translate a batch of review dictionaries
        
        Args:
            reviews: List of review dictionaries with 'sentence' key
            
        Returns:
            List of dictionaries with added 'translation' key
        """
        texts = [r.get('sentence', '') for r in reviews]
        translations = self.translate(texts)
        
        for review, translation in zip(reviews, translations):
            review['translation'] = translation
            
        return reviews