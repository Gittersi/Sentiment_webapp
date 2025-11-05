import streamlit as st
import requests
import time
from textblob import TextBlob
import re

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .sentiment-positive {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .sentiment-neutral {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def analyze_sentiment_local(text):
    """Local sentiment analysis using TextBlob and custom rules"""
    # TextBlob analysis
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Custom keyword analysis
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
        'love', 'best', 'awesome', 'brilliant', 'perfect', 'happy', 'joy',
        'beautiful', 'pleased', 'delighted', 'superb', 'outstanding', 'nice',
        'enjoy', 'satisfied', 'impressed', 'incredible', 'magnificent'
    ]
    
    negative_words = [
        'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate', 'poor',
        'disappointing', 'disappointed', 'sad', 'angry', 'upset', 'annoyed',
        'frustrated', 'useless', 'pathetic', 'disgusting', 'waste', 'failed',
        'dislike', 'unfair', 'wrong', 'broken', 'annoying'
    ]
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)
    
    # Combine TextBlob polarity with keyword counts
    if polarity > 0.1 or pos_count > neg_count:
        sentiment = "POSITIVE"
        confidence = min(abs(polarity) + (pos_count * 0.1), 0.99)
    elif polarity < -0.1 or neg_count > pos_count:
        sentiment = "NEGATIVE"
        confidence = min(abs(polarity) + (neg_count * 0.1), 0.99)
    else:
        sentiment = "NEUTRAL"
        confidence = 0.5 + abs(polarity)
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'positive_words': pos_count,
        'negative_words': neg_count
    }

def analyze_sentiment_huggingface(text):
    """Try to use Hugging Face API"""
    API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": text},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                results = data[0]
                # Find the highest confidence result
                top_result = max(results, key=lambda x: x['score'])
                return {
                    'sentiment': top_result['label'],
                    'confidence': top_result['score'],
                    'all_results': results,
                    'source': 'Hugging Face API'
                }
    except:
        pass
    
    return None

def get_emoji(sentiment):
    """Return emoji based on sentiment"""
    sentiment_upper = sentiment.upper()
    if 'POSITIVE' in sentiment_upper or 'POS' in sentiment_upper:
        return "😊"
    elif 'NEGATIVE' in sentiment_upper or 'NEG' in sentiment_upper:
        return "😢"
    else:
        return "😐"

def get_color(sentiment):
    """Return color based on sentiment"""
    sentiment_upper = sentiment.upper()
    if 'POSITIVE' in sentiment_upper or 'POS' in sentiment_upper:
        return "sentiment-positive"
    elif 'NEGATIVE' in sentiment_upper or 'NEG' in sentiment_upper:
        return "sentiment-negative"
    else:
        return "sentiment-neutral"

# Header
st.title("✨ Sentiment Analysis App")
st.markdown("### Analyze the emotional tone of any text using AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    analysis_method = st.radio(
        "Analysis Method",
        ["Auto (Try API, fallback to Local)", "Local Only", "API Only"],
        help="Choose how to analyze sentiment"
    )
    
    show_details = st.checkbox("Show Detailed Analysis", value=True)
    show_history = st.checkbox("Show Analysis History", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This app analyzes sentiment using:
    - **Hugging Face API**: State-of-the-art transformer models
    - **TextBlob**: Fast local analysis
    - **Custom Rules**: Keyword-based detection
    """)
    
    if st.button("Clear History"):
        st.session_state.analysis_history = []
        st.success("History cleared!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here...",
        help="Enter any text you want to analyze for sentiment"
    )
    
    # Example texts
    st.markdown("**Quick Examples:**")
    examples = [
        "I absolutely love this product! It's amazing and exceeded all my expectations!",
        "This is the worst experience I've ever had. Completely disappointed.",
        "It's okay, nothing special but not bad either. Just average.",
        "The customer service was fantastic! Very helpful and friendly staff.",
        "I'm so frustrated with this terrible quality. Would not recommend."
    ]
    
    example_cols = st.columns(len(examples))
    for idx, (col, example) in enumerate(zip(example_cols, examples)):
        with col:
            if st.button(f"Ex {idx+1}", key=f"ex_{idx}", use_container_width=True):
                text_input = example
                st.rerun()

with col2:
    st.markdown("### 📈 Quick Stats")
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)
        st.metric("Words", word_count)
        st.metric("Characters", char_count)
    else:
        st.info("Enter text to see statistics")

# Analyze button
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.error("⚠️ Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = None
            source = "Local Analysis"
            
            # Try different methods based on settings
            if analysis_method in ["Auto (Try API, fallback to Local)", "API Only"]:
                api_result = analyze_sentiment_huggingface(text_input)
                if api_result:
                    result = api_result
                    source = "Hugging Face API"
            
            # Fallback to local if API fails or local only selected
            if result is None and analysis_method != "API Only":
                local_result = analyze_sentiment_local(text_input)
                result = local_result
                source = "Local Analysis (TextBlob + Custom Rules)"
            
            if result is None:
                st.error("❌ API analysis failed. Please try 'Local Only' mode.")
            else:
                # Display results
                sentiment = result['sentiment']
                confidence = result['confidence']
                emoji = get_emoji(sentiment)
                
                st.markdown("---")
                st.markdown("## 📊 Analysis Results")
                
                # Main sentiment display
                st.markdown(f"""
                    <div class="{get_color(sentiment)}">
                        <h2 style="margin:0;">{emoji} {sentiment.upper()}</h2>
                        <p style="margin:5px 0 0 0; font-size: 18px;">Confidence: {confidence*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for confidence
                st.progress(confidence)
                
                # Detailed analysis
                if show_details:
                    st.markdown("### 📋 Detailed Analysis")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Sentiment", sentiment)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Confidence", f"{confidence*100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Source", source.split()[0])
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional metrics for local analysis
                    if 'polarity' in result:
                        st.markdown("#### Sentiment Metrics")
                        metric_col1, metric_col2 = st.columns(2)
                        
                        with metric_col1:
                            st.write(f"**Polarity:** {result['polarity']:.3f}")
                            st.caption("Range: -1 (negative) to +1 (positive)")
                        
                        with metric_col2:
                            st.write(f"**Subjectivity:** {result['subjectivity']:.3f}")
                            st.caption("Range: 0 (objective) to 1 (subjective)")
                        
                        if result['positive_words'] > 0 or result['negative_words'] > 0:
                            st.write(f"**Positive Keywords Found:** {result['positive_words']}")
                            st.write(f"**Negative Keywords Found:** {result['negative_words']}")
                
                # Add to history
                st.session_state.analysis_history.append({
                    'text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })

# Show history
if show_history and st.session_state.analysis_history:
    st.markdown("---")
    st.markdown("## 📜 Analysis History")
    
    for idx, item in enumerate(reversed(st.session_state.analysis_history[-10:])):
        with st.expander(f"{get_emoji(item['sentiment'])} {item['sentiment']} - {item['timestamp']}"):
            st.write(f"**Text:** {item['text']}")
            st.write(f"**Confidence:** {item['confidence']*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with Streamlit | Powered by Hugging Face & TextBlob</p>
        <p style="font-size: 12px;">💡 Tip: Use detailed mode to see comprehensive analysis metrics</p>
    </div>
""", unsafe_allow_html=True)