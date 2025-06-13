"""
Data loading utilities for Turkish product reviews dataset
"""
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, dataset_name="fthbrmnby/turkish_product_reviews", test_size=0.2):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.dataset = None
        self.df = None
        
    def load_dataset(self):
        """Load the Turkish product reviews dataset from Hugging Face"""
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name)
            
            # Convert to pandas DataFrame for easier manipulation
            self.df = pd.DataFrame(self.dataset['train'])
            
            # Map sentiment labels to readable format
            self.df['sentiment_label'] = self.df['sentiment'].map({
                0: 'negative',
                1: 'positive'
            })
            
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Sentiment distribution:\n{self.df['sentiment_label'].value_counts()}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_sample_data(self, n_samples=10):
        """Get a random sample of reviews"""
        if self.df is None:
            self.load_dataset()
        
        return self.df.sample(n=min(n_samples, len(self.df)))
    
    def prepare_train_test_split(self):
        """Split data into train and test sets"""
        if self.df is None:
            self.load_dataset()
            
        X = self.df['sentence'].values
        y = self.df['sentiment'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_statistics(self):
        """Get dataset statistics"""
        if self.df is None:
            self.load_dataset()
            
        stats = {
            'total_reviews': len(self.df),
            'positive_reviews': len(self.df[self.df['sentiment'] == 1]),
            'negative_reviews': len(self.df[self.df['sentiment'] == 0]),
            'avg_review_length': self.df['sentence'].str.len().mean(),
            'max_review_length': self.df['sentence'].str.len().max(),
            'min_review_length': self.df['sentence'].str.len().min()
        }
        
        return stats