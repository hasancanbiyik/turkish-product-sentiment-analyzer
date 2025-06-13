"""
Streamlit web application for Turkish Product Review Sentiment Analyzer
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer import TurkishProductSentimentAnalyzer
from src.data_loader import DataLoader

# Page configuration
st.set_page_config(
    page_title="Turkish Product Review Sentiment Analyzer",
    page_icon="🇹🇷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer():
    """Load the sentiment analyzer (cached)"""
    return TurkishProductSentimentAnalyzer()


@st.cache_data
def load_sample_data():
    """Load sample data from the dataset"""
    loader = DataLoader()
    df = loader.load_dataset()
    return loader.get_sample_data(100)


def display_sentiment_result(result):
    """Display a single sentiment analysis result"""
    sentiment = result["sentiment"]["label"]
    score = result["sentiment"]["score"]
    confidence = result["sentiment"]["confidence_level"]
    
    # Color based on sentiment
    if "positive" in sentiment.lower():
        sentiment_class = "sentiment-positive"
        emoji = "😊"
    else:
        sentiment_class = "sentiment-negative"
        emoji = "😔"
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f'<div class="{sentiment_class}">', unsafe_allow_html=True)
        st.markdown(f"**Original (Turkish):** {result['original_text']}")
        if "translation" in result:
            st.markdown(f"**Translation (English):** {result['translation']['text']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.metric("Sentiment", f"{emoji} {sentiment.title()}")
        st.metric("Confidence", f"{score:.2%}")
        st.metric("Level", confidence.title())


def main():
    # Header
    st.title("🇹🇷 Turkish Product Review Sentiment Analyzer")
    st.markdown("Analyze sentiment of Turkish product reviews with English translation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Review", "Batch Analysis", "Dataset Explorer"]
        )
        
        translate_option = st.checkbox("Translate to English", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info(
            "**Sentiment Model**: BERTurk\n\n"
            "**Translation Model**: Helsinki-NLP Opus MT"
        )
    
    # Initialize analyzer
    with st.spinner("Loading models..."):
        analyzer = load_analyzer()
    
    # Main content based on mode
    if analysis_mode == "Single Review":
        st.header("🔍 Analyze Single Review")
        
        # Input area
        review_text = st.text_area(
            "Enter a Turkish product review:",
            placeholder="Ürün çok güzel ve kaliteli. Hızlı kargo için teşekkürler!",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            analyze_button = st.button("🚀 Analyze", type="primary")
        with col2:
            if st.button("📋 Load Example"):
                review_text = "Bu ürünü çok beğendim. Kalitesi fiyatına göre gerçekten çok iyi. Kargo da hızlıydı, teşekkürler!"
                st.rerun()
        
        if analyze_button and review_text:
            with st.spinner("Analyzing..."):
                result = analyzer.analyze_single_review(review_text, translate=translate_option)
            
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            display_sentiment_result(result)
            
            # Detailed scores
            with st.expander("📈 Detailed Scores"):
                scores_df = pd.DataFrame([result["sentiment"]["all_scores"]])
                fig = px.bar(
                    scores_df.T.reset_index(),
                    x='index',
                    y=0,
                    labels={'index': 'Sentiment', 0: 'Score'},
                    title="Sentiment Probability Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Text statistics
            if "text_stats" in result:
                with st.expander("📝 Text Statistics"):
                    stats = result["text_stats"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", stats["word_count"])
                    with col2:
                        st.metric("Character Count", stats["char_count"])
                    with col3:
                        st.metric("Avg Word Length", f"{stats['avg_word_length']:.1f}")
    
    elif analysis_mode == "Batch Analysis":
        st.header("📚 Batch Analysis")
        
        # Text input for multiple reviews
        reviews_text = st.text_area(
            "Enter multiple reviews (one per line):",
            placeholder="Ürün harika!\nKötü bir deneyim oldu.\nFiyat performans ürünü.",
            height=200
        )
        
        if st.button("🚀 Analyze Batch", type="primary"):
            if reviews_text:
                reviews = [r.strip() for r in reviews_text.split('\n') if r.strip()]
                
                with st.spinner(f"Analyzing {len(reviews)} reviews..."):
                    results = analyzer.analyze_batch(reviews, translate=translate_option)
                
                st.markdown("---")
                st.subheader(f"📊 Analysis Results ({len(results)} reviews)")
                
                # Summary statistics
                summary = analyzer.get_sentiment_summary(reviews)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", summary["total_reviews"])
                with col2:
                    positive_pct = summary.get("positive_percentage", 0)
                    st.metric("Positive", f"{positive_pct:.1f}%")
                with col3:
                    negative_pct = summary.get("negative_percentage", 0)
                    st.metric("Negative", f"{negative_pct:.1f}%")
                with col4:
                    avg_conf = summary["average_confidence"]
                    st.metric("Avg Confidence", f"{avg_conf:.2%}")
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig = px.pie(
                        values=list(summary["sentiment_distribution"].values()),
                        names=list(summary["sentiment_distribution"].keys()),
                        title="Sentiment Distribution",
                        color_discrete_map={"positive": "#28a745", "negative": "#dc3545"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    confidence_scores = [r["sentiment"]["score"] for r in results]
                    fig = go.Figure(data=[go.Histogram(x=confidence_scores, nbinsx=20)])
                    fig.update_layout(
                        title="Confidence Score Distribution",
                        xaxis_title="Confidence Score",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                with st.expander("📋 Detailed Results"):
                    results_df = pd.DataFrame([
                        {
                            "Review": r["original_text"][:50] + "...",
                            "Sentiment": r["sentiment"]["label"],
                            "Confidence": f"{r['sentiment']['score']:.2%}",
                            "Translation": r.get("translation", "")[:50] + "..." if translate_option else "N/A"
                        }
                        for r in results
                    ])
                    st.dataframe(results_df, use_container_width=True)
    
    else:  # Dataset Explorer
        st.header("🗂️ Dataset Explorer")
        
        with st.spinner("Loading sample data..."):
            sample_df = load_sample_data()
        
        st.subheader("Sample Reviews from Dataset")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Size", len(sample_df))
        with col2:
            positive_count = len(sample_df[sample_df['sentiment'] == 1])
            st.metric("Positive Reviews", positive_count)
        with col3:
            negative_count = len(sample_df[sample_df['sentiment'] == 0])
            st.metric("Negative Reviews", negative_count)
        
        # Filter options
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All", "Positive", "Negative"]
        )
        
        if sentiment_filter == "Positive":
            filtered_df = sample_df[sample_df['sentiment'] == 1]
        elif sentiment_filter == "Negative":
            filtered_df = sample_df[sample_df['sentiment'] == 0]
        else:
            filtered_df = sample_df
        
        # Analyze sample
        if st.button("🔬 Analyze Sample", type="primary"):
            with st.spinner("Analyzing sample reviews..."):
                analyzed_df = analyzer.analyze_dataframe(
                    filtered_df.head(20),
                    translate=translate_option
                )
            
            st.subheader("Analysis Results")
            
            # Accuracy if ground truth available
            if 'sentiment' in analyzed_df.columns:
                analyzed_df['correct'] = (
                    (analyzed_df['sentiment'] == 1) & (analyzed_df['sentiment_prediction'] == 'positive') |
                    (analyzed_df['sentiment'] == 0) & (analyzed_df['sentiment_prediction'] == 'negative')
                )
                accuracy = analyzed_df['correct'].mean()
                st.success(f"Model Accuracy on Sample: {accuracy:.2%}")
            
            # Show results
            display_cols = ['sentence', 'sentiment_label', 'sentiment_prediction', 'sentiment_score']
            if translate_option:
                display_cols.append('english_translation')
            
            st.dataframe(
                analyzed_df[display_cols].head(20),
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, Hugging Face Transformers, and Turkish NLP models"
    )


if __name__ == "__main__":
    main()