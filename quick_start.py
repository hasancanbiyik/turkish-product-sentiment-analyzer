"""
Quick start script to demonstrate the Turkish Product Review Sentiment Analyzer
"""

from src.sentiment_analyzer import TurkishProductSentimentAnalyzer
from src.data_loader import DataLoader
import json


def main():
    print("🇹🇷 Turkish Product Review Sentiment Analyzer")
    print("=" * 50)
    
    # Initialize the analyzer
    print("\n📚 Loading models (this may take a moment on first run)...")
    analyzer = TurkishProductSentimentAnalyzer()
    print("✅ Models loaded successfully!")
    
    # Example reviews
    print("\n🔍 Analyzing sample reviews...")
    sample_reviews = [
        "Bu ürünü çok beğendim. Kalitesi harika, hızlı kargo için teşekkürler!",
        "Ürün açıklamasındaki gibi değil. Çok kötü kalite, paranıza yazık.",
        "Fiyatına göre idare eder. Çok da bir beklentim yoktu zaten.",
        "Mükemmel! Tam da aradığım ürün. Herkese tavsiye ederim.",
        "Kargo çok geç geldi ve ürün hasarlıydı. Hiç memnun kalmadım."
    ]
    
    results = []
    for i, review in enumerate(sample_reviews, 1):
        print(f"\n--- Review {i} ---")
        print(f"Turkish: {review}")
        
        # Analyze
        result = analyzer.analyze_single_review(review, translate=True)
        
        print(f"Sentiment: {result['sentiment']['label'].upper()} "
              f"(Confidence: {result['sentiment']['score']:.1%})")
        print(f"English: {result['translation']['text']}")
        
        results.append({
            "review_id": i,
            "original": review,
            "translation": result['translation']['text'],
            "sentiment": result['sentiment']['label'],
            "confidence": round(result['sentiment']['score'], 3)
        })
    
    # Save results
    print("\n💾 Saving results to 'examples/sample_results.json'...")
    import os
    os.makedirs('examples', exist_ok=True)
    
    with open('examples/sample_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ Results saved!")
    
    # Load and analyze dataset sample
    print("\n📊 Loading dataset sample...")
    loader = DataLoader()
    sample_df = loader.get_sample_data(20)
    
    # Get summary
    reviews_list = sample_df['sentence'].tolist()
    summary = analyzer.get_sentiment_summary(reviews_list)
    
    print(f"\n📈 Dataset Sample Analysis:")
    print(f"Total reviews analyzed: {summary['total_reviews']}")
    print(f"Positive: {summary.get('positive_percentage', 0):.1f}%")
    print(f"Negative: {summary.get('negative_percentage', 0):.1f}%")
    print(f"Average confidence: {summary['average_confidence']:.1%}")
    
    print("\n✨ Quick start complete! Run 'streamlit run app/app.py' for the web interface.")


if __name__ == "__main__":
    main()