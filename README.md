# 🇹🇷 Turkish Product Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](license_file.md)

An interactive Streamlit application for binary sentiment analysis of Turkish product reviews, with optional Turkish-to-English translation.

**Live demo:** [turkish-sentiment-analyzer-translator.streamlit.app](https://turkish-sentiment-analyzer-translator.streamlit.app/)

## ✨ Features

- **Single-review analysis:** Classify a Turkish review as positive or negative and display its confidence score.
- **Multi-review batch analysis:** Analyze reviews entered one per line and view aggregate sentiment metrics.
- **Turkish-to-English translation:** Translate reviews with the `Helsinki-NLP/opus-mt-tr-en` Opus-MT model.
- **Interactive visualizations:** Explore sentiment probabilities, class distribution, and confidence distribution with Plotly.
- **Dataset Explorer:** Sample and analyze reviews from the 235,165-row Turkish Product Reviews dataset.
- **Python API:** Analyze strings, lists of reviews, or pandas DataFrames programmatically.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/hasancanbiyik/turkish-product-sentiment-analyzer.git
cd turkish-product-sentiment-analyzer

# Install dependencies
python -m pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app/app.py
```

Open `http://localhost:8501` if Streamlit does not open it automatically. The first launch can take a few minutes while the pretrained models are downloaded and loaded.

## 🤖 Model and Data

### Sentiment model

The application loads the pretrained [`savasy/bert-base-turkish-sentiment-cased`](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased) binary classifier, which is based on BERTurk.

The upstream model card reports approximately **95.4% accuracy on its own evaluation set**. This repository does not retrain the classifier or independently reproduce that metric, so the upstream result should not be interpreted as a measured accuracy for this application.

### Dataset Explorer

The Dataset Explorer loads [`fthbrmnby/turkish_product_reviews`](https://huggingface.co/datasets/fthbrmnby/turkish_product_reviews), which contains **235,165 Turkish product reviews**. The dataset is used for exploration and sample analysis; it is not used to train the classifier in this repository.

## 🛠️ Tech Stack

- **Application:** Python, Streamlit
- **Sentiment analysis:** PyTorch, Hugging Face Transformers, BERTurk-based classifier
- **Translation:** Helsinki-NLP Opus-MT, SentencePiece
- **Data handling:** pandas, Hugging Face Datasets, scikit-learn
- **Visualization:** Plotly

## 📁 Project Structure

```text
turkish-product-sentiment-analyzer/
├── app/
│   └── app.py                 # Streamlit dashboard
├── config/
│   └── config.yaml            # Model, dataset, and app settings
├── examples/
│   └── sample_results.json    # Example output
├── src/
│   ├── data_loader.py         # Dataset loading and sampling
│   ├── model.py               # Sentiment model wrapper
│   ├── preprocessor.py        # Turkish text preprocessing
│   ├── sentiment_analyzer.py  # End-to-end analysis pipeline
│   └── translator.py          # Turkish-to-English translation
├── quick_start.py             # Command-line demonstration
├── requirements.txt
└── README.md
```

## 📖 Usage

### Analyze a single review

```python
from src.sentiment_analyzer import TurkishProductSentimentAnalyzer

analyzer = TurkishProductSentimentAnalyzer()
result = analyzer.analyze_single_review(
    "Bu ürün gerçekten harika!",
    translate=True,
)

print(result["sentiment"]["label"])
print(result["sentiment"]["score"])
print(result["translation"]["text"])
```

### Analyze multiple reviews

In the dashboard, select **Batch Analysis** and enter one review per line.

```python
reviews = [
    "Ürün harika!",
    "Kötü bir deneyim oldu.",
    "Fiyatına göre idare eder.",
]

results = analyzer.analyze_batch(reviews, translate=False)
```

### Analyze a CSV file programmatically

The dashboard does not currently include a CSV uploader. CSV files can be processed through the Python API when the review text is stored in a `sentence` column:

```python
import pandas as pd

reviews_df = pd.read_csv("reviews.csv")
results_df = analyzer.analyze_dataframe(
    reviews_df,
    text_column="sentence",
    translate=False,
)
results_df.to_csv("analyzed_reviews.csv", index=False)
```

## ☁️ Deployment

The application is deployed on Streamlit Community Cloud:

[Open the live application](https://turkish-sentiment-analyzer-translator.streamlit.app/)

To deploy your own copy, push the repository to GitHub, create an app at [share.streamlit.io](https://share.streamlit.io), and set the entry point to `app/app.py`.

## 📄 License

This project is available under the [MIT License](license_file.md).

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git switch -c feature-name`.
3. Commit your changes.
4. Push the branch to your fork.
5. Open a pull request.

## 📬 Contact

**Hasan Can Biyik**

- Email: hasanc.biyik@gmail.com
- LinkedIn: [linkedin.com/in/hasancanbyk](https://www.linkedin.com/in/hasancanbyk/)

---

⭐ Star the repository if you find it useful.
