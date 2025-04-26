# Twitter Sentiment Analysis with Twitter-RoBERTa and Streamlit

## Project Overview
This project provides a complete pipeline for training a Twitter sentiment analysis model using Twitter-RoBERTa in Google Colab and deploying it as a Streamlit web application. The model is specifically fine-tuned on Twitter data to better understand Twitter-specific language, slang, hashtags, and abbreviations.

## Deployment Link
🚀 Check out this awesome [Twitter Sentiment Analysis](https://anshika-twitter-sentiment-analysis.streamlit.app/) tool that lets you explore how people feel about any topic in real-time!



## Project Structure
```
Twitter-Sentiment-analysis/
├── Notebook/
│   ├── train_sentiment_model.ipynb  # Colab notebook for model training
│   └── test_sentiment_model.ipynb   # Notebook for model evaluation
├── models/
│   └── sentiment_model/            # Trained model files
├── src/
│   ├── app.py                      # Streamlit web application
│   ├── model.py                    # Model loading and prediction
│   └── data_processing.py          # Text preprocessing utilities
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Dataset
- Training dataset: CSV with columns (Tweet ID, entity, sentiment, Tweet content)
- Test dataset: Same structure as training data
- Sentiment labels: Positive (2), Negative (0), Neutral (1), Irrelevant (excluded)
- The model handles headerless CSV files by assigning appropriate column names

## Setup Instructions

### 1. Environment Setup (Recommended: uv)

This project uses [uv](https://github.com/astral-sh/uv) as the package manager for fast, modern Python dependency management.

#### Create and activate a virtual environment:
```bash
# Create a new virtual environment named .venv
euv venv .venv

# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Activate the virtual environment (Linux/Mac)
source .venv/bin/activate
```

#### Install all dependencies with uv:
```bash
uv pip install -r requirements.txt
```

*If you don't have uv, install it from https://github.com/astral-sh/uv or with `pip install uv`.*

### 2. Training the Model (Google Colab)
1. Upload your training data CSV to Colab
2. Open `train_sentiment_model.ipynb`
3. Update the file path in the notebook
4. Set `has_headers=False` if your CSV doesn't have headers
5. Run all cells
6. The notebook includes memory optimizations for Colab's T4 GPU
7. Download the `sentiment_model.zip` file

### 2. Testing the Model (Google Colab)
1. Upload your test data CSV to Colab
2. Open `test_sentiment_model.ipynb`
3. Update the file path in the notebook
4. Run all cells to see evaluation metrics

### 3. Running the Streamlit App Locally
1. Make sure your virtual environment is activated (see above)
2. Place the unzipped `sentiment_model` folder in the `models/` directory
3. Run the app:
   ```bash
   streamlit run src/app.py
   ```
4. The app will load the model from the `models/sentiment_model` directory and launch with a modern dark UI.

## Deployment
### Streamlit Cloud
1. Create a GitHub repository with:
   - `src/app.py`
   - `models/sentiment_model` folder
   - `requirements.txt` (generated from your environment)
2. Connect your GitHub to Streamlit Cloud
3. Deploy the app

## Package Management
- **uv** is used for dependency management and installation. It's much faster than pip and supports modern workflows.
- All dependencies are listed in `requirements.txt`.
- For new packages, use:
  ```bash
  uv pip install <package-name>
  ```
- To update requirements.txt after adding/removing packages:
  ```bash
  uv pip freeze > requirements.txt
  ```

## Model Details
- **Base Model**: Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)
- **Advantages**:
  - Pre-trained on Twitter data for better understanding of Twitter language
  - Handles Twitter-specific features like hashtags, mentions, and emojis
  - Fine-tuned for 3-class sentiment classification

## Memory Optimizations
- Dynamic batch sizing based on available GPU memory
- Gradient accumulation for effective large batch training
- Mixed precision (FP16) training
- Gradient checkpointing to save memory

## Maintenance & Enhancements
1. **Model Retraining**: Periodically retrain with new data
2. **Performance Improvements**:
   - Try different models (DistilRoBERTa for faster inference)
   - Add hyperparameter tuning
3. **UI Enhancements**:
   - Add batch processing for multiple tweets
   - Include visualizations of sentiment trends

## Troubleshooting
- **Memory errors**: Reduce batch size in the training notebook if you run out of memory on Colab
- **Model not found**: Ensure `models/sentiment_model` folder exists and contains the correct files
- **CSV format issues**: Set `has_headers=False` in the notebook if your CSV doesn't have headers
- **uv not found**: Install uv from https://github.com/astral-sh/uv or use `pip install uv` as a fallback

---

**Enjoy your modern, fast, and beautiful Twitter sentiment analysis app!**

For any issues, please open an issue on the repository or contact the maintainer.
