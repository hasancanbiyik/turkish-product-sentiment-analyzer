"""
Text preprocessing utilities for Turkish text
"""
import re
import string
from typing import List, Union


class TurkishTextPreprocessor:
    def __init__(self):
        # Turkish specific characters
        self.turkish_chars = "çğıöşüÇĞİÖŞÜ"
        
        # Common Turkish stop words (minimal list for sentiment analysis)
        # We keep it minimal because some stopwords might be important for sentiment
        self.stopwords = {
            've', 'ile', 'ki', 'da', 'de', 'mi', 'mu', 'mı', 'mü'
        }
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning while preserving Turkish characters"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove HTML tags if any
        text = re.sub(r'<.*?>', '', text)
        
        return text.strip()
    
    def normalize_turkish(self, text: str) -> str:
        """Normalize Turkish-specific issues"""
        if not text:
            return ""
            
        # Fix common Turkish character encoding issues
        replacements = {
            'ý': 'ı',
            'ð': 'ğ',
            'þ': 'ş',
            'Ý': 'İ',
            'Ð': 'Ğ',
            'Þ': 'Ş'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
    
    def remove_excessive_punctuation(self, text: str) -> str:
        """Remove excessive punctuation while keeping sentiment indicators"""
        # Keep single punctuation marks but remove repetitions
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove excessive dots
        text = re.sub(r'\.{2,}', '.', text)
        
        return text
    
    def preprocess(self, text: Union[str, List[str]], 
                  remove_stopwords: bool = False,
                  keep_punctuation: bool = True) -> Union[str, List[str]]:
        """
        Full preprocessing pipeline
        
        Args:
            text: Single text or list of texts
            remove_stopwords: Whether to remove stopwords (default: False for sentiment)
            keep_punctuation: Whether to keep punctuation (default: True for sentiment)
        """
        if isinstance(text, list):
            return [self.preprocess(t, remove_stopwords, keep_punctuation) for t in text]
        
        # Apply cleaning steps
        text = self.clean_text(text)
        text = self.normalize_turkish(text)
        text = self.remove_excessive_punctuation(text)
        
        # Optional: remove stopwords
        if remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)
        
        # Optional: remove all punctuation
        if not keep_punctuation:
            # Keep Turkish characters but remove other punctuation
            allowed_chars = string.ascii_letters + self.turkish_chars + ' '
            text = ''.join(c for c in text if c in allowed_chars)
        
        return text.strip()
    
    def get_text_stats(self, text: str) -> dict:
        """Get statistics about the text"""
        words = text.split()
        return {
            'char_count': len(text),
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }