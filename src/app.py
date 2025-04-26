# <xaiArtifact artifact_id="7fca53d3-eeb9-4e51-9992-096280baf133" title="app.py" contentType="text/x-python">
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import datetime
import json
from matplotlib.patches import Patch

# Fix for PyTorch compatibility issue with Streamlit
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    .css-18e3th9 {
        background-color: #121212;
        color: #f0f0f0;
    }
    .css-1d391kg {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #f0f0f0;
    }
    .stTextArea > div > div > textarea {
        background-color: #2d2d2d;
        color: #f0f0f0;
    }
    .stSelectbox > div > div > div {
        background-color: #2d2d2d;
        color: #f0f0f0;
    }
    .main-header {
        background-color: #1A1A1A;
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid #1DA1F2;
    }
    .sentiment-card {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #1DA1F2;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #0c85d0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .history-item {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .history-item:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model():
    # Only use local model
    with st.spinner("Loading model... This may take a moment."):
        tokenizer = AutoTokenizer.from_pretrained('models/sentiment_model')
        model = AutoModelForSequenceClassification.from_pretrained('models/sentiment_model')
    return model, tokenizer

# Function to save analysis history
def save_to_history(tweet, sentiment, probabilities):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Load existing history or create new
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Add new entry (limit to last 10)
    st.session_state.history.insert(0, {
        "timestamp": timestamp,
        "tweet": tweet,
        "sentiment": sentiment,
        "probabilities": probabilities.tolist()
    })
    
    # Keep only the 10 most recent entries
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]

# Function to create donut chart for sentiment distribution
def create_sentiment_distribution(history_data):
    if not history_data:
        return None
        
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for item in history_data:
        sentiment_counts[item['sentiment']] += 1
    
    # Create donut chart
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ['#4ecdc4', '#f9d423', '#ff6b6b']
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [sentiment_counts[label] for label in labels]
    
    # Create pie chart with a hole in the middle
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=None,
        colors=colors, 
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='w')
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Add legend
    legend_elements = [Patch(facecolor=colors[i], label=f"{labels[i]} ({sizes[i]})") 
                      for i in range(len(labels))]
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.5))
    
    plt.tight_layout()
    return fig

# Predict sentiment
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
    predicted_class = np.argmax(probabilities)
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_map[predicted_class], probabilities

# Main app
def main():
    # Initialize session state for theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🐦 Settings</h2>", unsafe_allow_html=True)
        
        # Theme selection
        theme_options = ['Light', 'Dark', 'Twitter Blue']
        selected_theme = st.selectbox("Select Theme", theme_options, index=0)
        st.session_state.theme = selected_theme.lower()
        
        # About section
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>About</h3>", unsafe_allow_html=True)
        st.markdown("""
        This app uses a custom-trained Twitter-RoBERTa model for sentiment analysis of tweets. 
        The model was fine-tuned on Twitter data to better understand Twitter-specific language patterns.
        
        **Model Features:**
        - Pre-trained on Twitter data
        - Fine-tuned for 3-class sentiment analysis
        - Handles hashtags, mentions, and emojis
        """)
        
        # History section
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Analysis History</h3>", unsafe_allow_html=True)
        
        if 'history' in st.session_state and st.session_state.history:
            # Show sentiment distribution chart
            st.subheader("Sentiment Distribution")
            dist_chart = create_sentiment_distribution(st.session_state.history)
            if dist_chart:
                st.pyplot(dist_chart)
            
            # Show history items
            st.subheader("Recent Analyses")
            for i, item in enumerate(st.session_state.history):
                sentiment_colors = {
                    'Negative': '#ff6b6b',
                    'Neutral': '#f9d423',
                    'Positive': '#4ecdc4'
                }
                
                # Truncate tweet if too long
                tweet_display = item['tweet'] if len(item['tweet']) < 50 else item['tweet'][:47] + "..."
                
                # Create clickable history item
                st.markdown(f"""
                <div class='history-item' style='background-color: {sentiment_colors[item['sentiment']]}20; border-left: 4px solid {sentiment_colors[item['sentiment']]}'>                    
                    <small>{item['timestamp']}</small><br>
                    <strong>{item['sentiment']}</strong>: {tweet_display}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No analysis history yet. Analyze some tweets to see history.")
    
    # Main content - Header with logo and title in a styled container
    st.markdown("""
    <div class='main-header'>        
        <div style='display: flex; align-items: center;'>            
            <div style='font-size:3.5rem; margin-right: 15px;'>🐦</div>
            <div>
                <h1 style='margin: 0; padding: 0;'>Twitter Sentiment Analysis</h1>
                <p style='margin: 0; padding: 0;'>Analyze sentiment of tweets using Custom-Trained Twitter-RoBERTa model</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model info box with better styling
    st.markdown("""
    <div style='background-color: #333333; color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='margin-top:0'>📊 Model Information</h3>
        <p>This application uses a fine-tuned <b>Twitter-RoBERTa</b> model specifically trained on Twitter data for accurate sentiment analysis.</p>
        <p>The model classifies tweets into three sentiment categories: <span style='color:#ff6b6b'>Negative</span>, <span style='color:#f9d423'>Neutral</span>, and <span style='color:#4ecdc4'>Positive</span>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except:
        st.error("Model not found. Please ensure 'models/sentiment_model' folder exists.")
        return
    
    # Create two columns for input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input text with styled container
        st.markdown("<h3 style='margin-bottom:10px'>Tweet Input</h3>", unsafe_allow_html=True)
        tweet_text = st.text_area("Enter a tweet to analyze:", height=120)
    
    with col2:
        # Example tweets in a styled container
        st.markdown("<h3 style='margin-bottom:10px'>Examples</h3>", unsafe_allow_html=True)
        example_tweets = [
            "I love this product! It's amazing! 😍",
            "This is just okay, nothing special.",
            "Terrible experience, would not recommend. 😡",
            "The weather is nice today. ☀️",
            "I'm frustrated with the service. 😤"
        ]
        
        # Example selection with better styling
        example = st.selectbox("Select an example:", ["None"] + example_tweets)
        if example != "None":
            tweet_text = example
    
    # Analyze button with better styling
    analyze_button = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_button:
        if not tweet_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            # Add a progress bar for better UX
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate processing time
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            with st.spinner("Finalizing analysis..."):
                sentiment, probabilities = predict_sentiment(tweet_text, model, tokenizer)
                # Save to history
                save_to_history(tweet_text, sentiment, probabilities)
            
            # Remove progress bar after completion
            progress_bar.empty()
            
            # Success message with animation
            st.success("Analysis completed successfully!")
            
            # Results section with columns
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Analysis Results</h2>", unsafe_allow_html=True)
            
            # Create three columns for results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display sentiment with color coding
                sentiment_colors = {
                    'Negative': '#ff6b6b',
                    'Neutral': '#f9d423',
                    'Positive': '#4ecdc4'
                }
                
                sentiment_emojis = {
                    'Negative': '😞',
                    'Neutral': '😐',
                    'Positive': '😊'
                }
                
                st.markdown(f"""
                <div style='background-color: #333333; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h1 style='font-size: 3rem; margin-bottom: 10px;'>{sentiment_emojis[sentiment]}</h1>
                    <h2 style='color: {sentiment_colors[sentiment]}; margin-top: 0;'>{sentiment}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create confidence score visualization
                fig, ax = plt.subplots(figsize=(8, 2.5))
                
                # Define colors for each sentiment
                colors = ['#ff6b6b', '#f9d423', '#4ecdc4']
                labels = ['Negative', 'Neutral', 'Positive']
                
                # Create horizontal bar chart
                bars = ax.barh(labels, probabilities, color=colors)
                
                # Add percentage labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                            va='center', fontweight='bold')
                
                # Customize chart
                ax.set_xlim(0, 1.1)
                ax.set_title('Confidence Scores', fontsize=14, fontweight='bold')
                ax.set_facecolor('#2d2d2d')
                fig.patch.set_facecolor('#2d2d2d')
                ax.tick_params(colors='#f0f0f0')
                ax.xaxis.label.set_color('#f0f0f0')
                ax.yaxis.label.set_color('#f0f0f0')
                ax.title.set_color('#f0f0f0')
                
                # Display the plot
                st.pyplot(fig)
                
                # Add tweet text display with word highlighting
                words = tweet_text.split()
                highlighted_text = tweet_text
                
                # Highlight positive and negative words (simplified approach)
                positive_words = ["love", "great", "good", "amazing", "excellent", "happy", "best", "awesome"]
                negative_words = ["hate", "bad", "terrible", "awful", "worst", "horrible", "disappointed", "frustrated"]
                
                # Create highlighted version
                highlighted_html = ""
                for word in words:
                    clean_word = word.lower().strip(".,!?;:()[]{}'\"").strip()
                    if clean_word in positive_words:
                        highlighted_html += f"<span style='background-color: rgba(78, 205, 196, 0.3); padding: 2px; border-radius: 3px;'>{word}</span> "
                    elif clean_word in negative_words:
                        highlighted_html += f"<span style='background-color: rgba(255, 107, 107, 0.3); padding: 2px; border-radius: 3px;'>{word}</span> "
                    else:
                        highlighted_html += f"{word} "
                
                st.markdown(f"""
                <div style='background-color: #2d2d2d; color: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 10px; border: 1px solid #444444;'>
                    <h4 style='margin-top: 0;'>Analyzed Text</h4>
                    <p>{highlighted_html}</p>
                    <div style='display: flex; margin-top: 10px; font-size: 0.8em;'>
                        <div style='margin-right: 15px;'><span style='background-color: rgba(78, 205, 196, 0.3); padding: 2px; border-radius: 3px;'>Positive words</span></div>
                        <div><span style='background-color: rgba(255, 107, 107, 0.3); padding: 2px; border-radius: 3px;'>Negative words</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a "Share Results" button (simulated)
                share_col1, share_col2 = st.columns([1, 1])
                with share_col1:
                    if st.button("📋 Copy Results", use_container_width=True):
                        st.info("Results copied to clipboard (simulated)")
                with share_col2:
                    if st.button("📊 Download Report", use_container_width=True):
                        st.info("Report download started (simulated)")

if __name__ == "__main__":
    main()