# 🇹🇷 Turkish Product Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

> Real-time sentiment analysis for Turkish product reviews with English translation support

## ✨ Features

- **🔍 Real-time Analysis**: Classify Turkish product reviews as positive/negative
- **🌍 Translation**: Automatic English translation of Turkish reviews  
- **📊 Visualizations**: Interactive charts and sentiment metrics
- **📁 Batch Processing**: Upload CSV files for bulk analysis
- **🎯 High Accuracy**: Uses BERTurk model trained on 235K Turkish reviews

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/hasancanbiyik/turkish-product-sentiment-analyzer.git
cd turkish-product-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run quick_start.py
```

Open `http://localhost:8501` in your browser.

## 📊 Model Performance

- **Dataset**: 235,165 Turkish product reviews
- **Accuracy**: 94.2%
- **Model**: BERTurk (dbmdz/bert-base-turkish-cased)
- **Languages**: Turkish (primary) + English translation

## 🛠️ Tech Stack

- **Backend**: Python, PyTorch, Transformers
- **Frontend**: Streamlit
- **NLP**: BERTurk, NLTK, spaCy
- **Translation**: Google Translate API
- **Visualization**: Plotly, Matplotlib

## 📁 Project Structure

```
turkish-product-sentiment-analyzer/
├── src/                    # Core modules
│   ├── sentiment_analyzer.py
│   ├── preprocessor.py
│   ├── translator.py
│   └── model.py
├── app/                    # Streamlit app
├── config/                 # Configuration files
├── examples/               # Sample data
├── quick_start.py          # Main app file
├── requirements.txt        # Dependencies
└── README.md
```

## 📖 Usage

### Single Review Analysis
```python
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("Bu ürün gerçekten harika!")
print(result)  # {'sentiment': 'POSITIVE', 'confidence': 0.94}
```

### Batch Analysis
Upload a CSV file with `review_text` column through the web interface.

## 🎯 Examples

**Turkish Input**: "Bu telefon çok kaliteli, herkese tavsiye ederim!"  
**Output**: Positive (95.3% confidence)  
**Translation**: "This phone is very high quality, I recommend it to everyone!"

**Turkish Input**: "Ürün beklentimin altında kaldı, kalitesi çok düşük"  
**Output**: Negative (91.7% confidence)  
**Translation**: "The product fell short of my expectations, very poor quality"

## 🚀 Deployment

Deploy on Streamlit Cloud:
1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Deploy with one click

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📧 Contact

**Hasan Can Biyik** - [GitHub](https://github.com/hasancanbiyik)

---

⭐ **Star this repo if you find it helpful!**