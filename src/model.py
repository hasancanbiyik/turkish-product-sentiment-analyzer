"""
Sentiment analysis model utilities
"""
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import logging
from typing import List, Union, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, model_name="savasy/bert-base-turkish-sentiment-cased"):
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Model-specific configurations
        self.model_configs = {
            "savasy/bert-base-turkish-sentiment-cased": {
                "labels": ["negative", "positive"],
                "label_map": {0: "negative", 1: "positive"}
            },
            "nlptown/bert-base-multilingual-uncased-sentiment": {
                "labels": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
                "label_map": {0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}
            }
        }
        
    def load_model(self):
        """Load the sentiment analysis model"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            logger.info("Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            raise
    
    def analyze(self, text: Union[str, List[str]], 
               return_all_scores: bool = False) -> Union[Dict, List[Dict]]:
        """
        Analyze sentiment of text(s)
        
        Args:
            text: Single text or list of texts
            return_all_scores: Whether to return all class probabilities
            
        Returns:
            Sentiment analysis results
        """
        if self.pipeline is None:
            self.load_model()
        
        single_text = isinstance(text, str)
        if single_text:
            text = [text]
        
        try:
            # Get predictions
            if return_all_scores:
                # Get probabilities for all classes
                outputs = self.model(**self.tokenizer(text, padding=True, truncation=True, return_tensors="pt"))
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                results = []
                for i, pred in enumerate(predictions):
                    scores = pred.detach().numpy()
                    
                    # Get label configuration
                    config = self.model_configs.get(self.model_name, {})
                    labels = config.get("labels", [f"LABEL_{j}" for j in range(len(scores))])
                    
                    result = {
                        "text": text[i],
                        "scores": {label: float(score) for label, score in zip(labels, scores)},
                        "label": labels[np.argmax(scores)],
                        "score": float(np.max(scores))
                    }
                    results.append(result)
            else:
                # Use pipeline for simple predictions
                predictions = self.pipeline(text, truncation=True, max_length=512)
                
                results = []
                for i, pred in enumerate(predictions):
                    result = {
                        "text": text[i],
                        "label": pred["label"],
                        "score": pred["score"]
                    }
                    results.append(result)
            
            return results[0] if single_text else results
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise
    
    def analyze_with_confidence(self, text: str) -> Dict:
        """
        Analyze sentiment with confidence scores and interpretation
        """
        result = self.analyze(text, return_all_scores=True)
        
        # Add confidence interpretation
        confidence = result["score"]
        if confidence > 0.9:
            confidence_level = "very high"
        elif confidence > 0.7:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
        
        result["confidence_level"] = confidence_level
        result["interpretation"] = self._get_interpretation(result["label"], confidence)
        
        return result
    
    def _get_interpretation(self, label: str, confidence: float) -> str:
        """Get human-readable interpretation of results"""
        if "positive" in label.lower() or "5 stars" in label:
            if confidence > 0.8:
                return "Highly positive review - customer is very satisfied"
            else:
                return "Positive review - customer is satisfied"
        elif "negative" in label.lower() or "1 star" in label:
            if confidence > 0.8:
                return "Highly negative review - customer is very dissatisfied"
            else:
                return "Negative review - customer is dissatisfied"
        else:
            return "Neutral review - mixed feelings"
    
    def compare_models(self, text: str, models: List[str]) -> Dict:
        """
        Compare predictions from multiple models
        
        Args:
            text: Text to analyze
            models: List of model names to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for model_name in models:
            try:
                # Create temporary analyzer for each model
                analyzer = SentimentAnalyzer(model_name)
                analyzer.load_model()
                
                result = analyzer.analyze_with_confidence(text)
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Error with model {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        return results