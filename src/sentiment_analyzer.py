"""
Main sentiment analysis module combining all components
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import logging
from .preprocessor import TurkishTextPreprocessor
from .model import SentimentAnalyzer
from .translator import TurkishEnglishTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TurkishProductSentimentAnalyzer:
    def __init__(self, 
                 sentiment_model="savasy/bert-base-turkish-sentiment-cased",
                 translation_model="Helsinki-NLP/opus-mt-tr-en",
                 preprocess_text=True):
        """
        Initialize the complete sentiment analysis pipeline
        
        Args:
            sentiment_model: Model name for sentiment analysis
            translation_model: Model name for translation
            preprocess_text: Whether to preprocess text before analysis
        """
        self.preprocessor = TurkishTextPreprocessor() if preprocess_text else None
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model)
        self.translator = TurkishEnglishTranslator(translation_model)
        
        # Load models
        logger.info("Initializing models...")
        self.sentiment_analyzer.load_model()
        self.translator.load_model()
        logger.info("All models loaded successfully!")
        
    def analyze_single_review(self, review: str, translate: bool = True) -> Dict:
        """
        Analyze a single product review
        
        Args:
            review: Turkish product review text
            translate: Whether to translate the review to English
            
        Returns:
            Dictionary with analysis results
        """
        # Preprocess if enabled
        processed_text = review
        if self.preprocessor:
            processed_text = self.preprocessor.preprocess(review)
        
        # Sentiment analysis
        sentiment_result = self.sentiment_analyzer.analyze_with_confidence(processed_text)
        
        # Translation
        translation_result = None
        if translate:
            translation_result = self.translator.translate_with_alignment(review)
        
        # Combine results
        result = {
            "original_text": review,
            "processed_text": processed_text,
            "sentiment": {
                "label": sentiment_result["label"],
                "score": sentiment_result["score"],
                "confidence_level": sentiment_result["confidence_level"],
                "interpretation": sentiment_result["interpretation"],
                "all_scores": sentiment_result.get("scores", {})
            }
        }
        
        if translation_result:
            result["translation"] = {
                "text": translation_result["translation"],
                "original_words": translation_result["original_length"],
                "translated_words": translation_result["translation_length"]
            }
        
        # Add text statistics
        if self.preprocessor:
            result["text_stats"] = self.preprocessor.get_text_stats(processed_text)
        
        return result
    
    def analyze_batch(self, reviews: List[str], 
                     translate: bool = True,
                     batch_size: int = 32) -> List[Dict]:
        """
        Analyze multiple reviews in batch
        
        Args:
            reviews: List of review texts
            translate: Whether to translate reviews
            batch_size: Batch size for processing
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            
            # Preprocess batch
            if self.preprocessor:
                processed_batch = self.preprocessor.preprocess(batch)
            else:
                processed_batch = batch
            
            # Sentiment analysis
            sentiment_results = self.sentiment_analyzer.analyze(processed_batch, return_all_scores=True)
            
            # Translation
            if translate:
                translations = self.translator.translate(batch)
            else:
                translations = [None] * len(batch)
            
            # Combine results
            for j, (original, processed, sentiment, translation) in enumerate(
                zip(batch, processed_batch, sentiment_results, translations)
            ):
                result = {
                    "id": i + j,
                    "original_text": original,
                    "processed_text": processed,
                    "sentiment": {
                        "label": sentiment["label"],
                        "score": sentiment["score"],
                        "all_scores": sentiment.get("scores", {})
                    }
                }
                
                if translation:
                    result["translation"] = translation
                
                results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                         text_column: str = "sentence",
                         translate: bool = True) -> pd.DataFrame:
        """
        Analyze reviews in a pandas DataFrame
        
        Args:
            df: DataFrame containing reviews
            text_column: Column name containing review text
            translate: Whether to translate reviews
            
        Returns:
            DataFrame with added sentiment analysis columns
        """
        reviews = df[text_column].tolist()
        results = self.analyze_batch(reviews, translate=translate)
        
        # Add results to dataframe
        df["sentiment_prediction"] = [r["sentiment"]["label"] for r in results]
        df["sentiment_score"] = [r["sentiment"]["score"] for r in results]
        
        if translate:
            df["english_translation"] = [r.get("translation", "") for r in results]
        
        # Add confidence scores if available
        if "scores" in results[0]["sentiment"]:
            for label in results[0]["sentiment"]["scores"].keys():
                df[f"score_{label}"] = [r["sentiment"]["scores"][label] for r in results]
        
        return df
    
    def get_sentiment_summary(self, reviews: List[str]) -> Dict:
        """
        Get summary statistics for a list of reviews
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with summary statistics
        """
        results = self.analyze_batch(reviews, translate=False)
        
        # Calculate statistics
        sentiments = [r["sentiment"]["label"] for r in results]
        scores = [r["sentiment"]["score"] for r in results]
        
        summary = {
            "total_reviews": len(reviews),
            "sentiment_distribution": pd.Series(sentiments).value_counts().to_dict(),
            "average_confidence": np.mean(scores),
            "confidence_stats": {
                "min": np.min(scores),
                "max": np.max(scores),
                "std": np.std(scores)
            }
        }
        
        # Calculate percentage distribution
        for sentiment in summary["sentiment_distribution"]:
            percentage = (summary["sentiment_distribution"][sentiment] / len(reviews)) * 100
            summary[f"{sentiment}_percentage"] = round(percentage, 2)
        
        return summary