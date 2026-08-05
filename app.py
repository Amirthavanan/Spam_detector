import re
import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------- STOP WORDS --------------------
stop_words = ENGLISH_STOP_WORDS

# -------------------- MODEL PATHS --------------------
MODEL_PATH = "logistic_regression_model.joblib"
VEC_PATH = "tfidf_vectorizer.joblib"

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer

model, tfidf = load_model()

# -------------------- PREPROCESS --------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)

    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------- PREDICT --------------------
def predict(message):
    processed = preprocess(message)
    vector = tfidf.transform([processed])

    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    if prediction == 1:
        return "Spam", probability[1]
    else:
        return "Ham", probability[0]

# -------------------- CSS --------------------
st.markdown("""
<style>

html, body, [data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#0b1020,#18233d,#090d17);
font-family:Arial,sans-serif;
color:white;
}

[data-testid="stHeader"]{
background:transparent;
}

.block-container{
max-width:1000px;
padding-top:1rem;
}

.hero{
padding:35px;
border-radius:25px;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(15px);
border:1px solid rgba(255,255,255,0.15);
text-align:center;
margin-bottom:25px;
}

.metric{
background:rgba(255,255,255,0.08);
padding:20px;
border-radius:18px;
text-align:center;
}

div.stButton > button{
width:100%;
height:60px;
border:none;
border-radius:35px;
font-size:20px;
font-weight:bold;
background:linear-gradient(90deg,#7c3aed,#2563eb);
color:white;
}

.result{
padding:25px;
border-radius:18px;
font-size:22px;
font-weight:bold;
text-align:center;
margin-top:20px;
}

.safe{
background:#133d22;
border:2px solid #22c55e;
}

.spam{
background:#4b1111;
border:2px solid #ef4444;
}

.footer{
text-align:center;
color:#bdbdbd;
margin-top:30px;
font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
<h1>🛡️ AI Spam Detector</h1>
<p>SMS Spam Detection using Logistic Regression & TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# -------------------- METRICS --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric">
    <h2>97%</h2>
    <p>Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric">
    <h2>TF-IDF</h2>
    <p>Vectorizer</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric">
    <h2>Logistic Regression</h2>
    <p>ML Model</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- INPUT --------------------
message = st.text_area(
    "📨 Enter SMS Message",
    height=180,
    placeholder="Type or paste your SMS here..."
)

# -------------------- BUTTON --------------------
if st.button("🚀 Detect Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        with st.spinner("Analyzing message..."):
            label, confidence = predict(message)

        if label == "Spam":

            st.markdown(
                f"""
                <div class="result spam">
                ⚠️ SPAM MESSAGE
                <br><br>
                Confidence : {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result safe">
                ✅ SAFE MESSAGE
                <br><br>
                Confidence : {confidence:.2%}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(confidence)

# -------------------- FOOTER --------------------
st.markdown("""
<div class="footer">
Streamlit | Logistic Regression | Scikit-Learn | Amirthavanan
</div>
""", unsafe_allow_html=True)
