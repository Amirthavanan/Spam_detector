import re
import joblib
import nltk
import streamlit as st
from nltk.corpus import stopwords
import nltk

resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4"
]
st.set_page_config(
    page_title="Spam detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

for pkg, path in [
    ("punkt","tokenizers/punkt"),
    ("stopwords","corpora/stopwords")
]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

stop_words = set(stopwords.words("english"))

MODEL_PATH="best_logistic_regression_model.pkl"
VEC_PATH="tfidf_vectorizer.pkl"

@st.cache_resource
def load():
    return joblib.load(MODEL_PATH), joblib.load(VEC_PATH)

model, tfidf = load()

def preprocess(txt):
    txt = re.sub(r"[^a-zA-Z\s]"," ",txt)
    txt = " ".join(
        w for w in nltk.word_tokenize(txt)
        if w.lower() not in stop_words
    )
    return txt

def predict(msg):
    x = tfidf.transform([preprocess(msg)])
    pred = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    return ("Spam", float(proba[1])) if pred == 1 else ("Ham", float(proba[0]))

st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css">
<style>
html,body,[data-testid="stAppViewContainer"]{
background:linear-gradient(135deg,#0b1020,#18233d,#090d17);
font-family:Poppins,sans-serif;
color:white;
}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent;}
.block-container{padding-top:1rem;max-width:1000px;}
.hero{
padding:35px;border-radius:28px;
background:rgba(255,255,255,.08);
backdrop-filter:blur(18px);
border:1px solid rgba(255,255,255,.15);
box-shadow:0 10px 40px rgba(0,0,0,.35);
text-align:center;
margin-bottom:25px;
}
.metric{
background:rgba(255,255,255,.07);
padding:18px;border-radius:18px;text-align:center;
}
div.stButton>button{
width:100%;height:58px;border-radius:40px;
background:linear-gradient(90deg,#7c3aed,#2563eb);
color:white;font-weight:700;font-size:18px;border:none;
}
.result{
padding:25px;border-radius:20px;margin-top:20px;
font-size:22px;font-weight:600;text-align:center;
}
.safe{background:#163a23;border:1px solid #22c55e;}
.spam{background:#4a1111;border:1px solid #ef4444;}
.footer{text-align:center;color:#bbb;padding-top:30px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1><i class="fa-solid fa-shield-halved"></i> SPAM DETECTOR </h1>
<p>Modern AI-powered SMS Spam Detection using Logistic Regression + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric"><h3>98%</h3><small>Accuracy</small></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric"><h3>TF-IDF</h3><small>Vectorizer</small></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric"><h3>Log Regression</h3><small>Model</small></div>', unsafe_allow_html=True)

msg = st.text_area("📨 Enter SMS Message", height=180)

if st.button("🚀 Detect Message"):
    if msg.strip():
        with st.spinner("Analyzing..."):
            label, conf = predict(msg)
        css = "spam" if label == "Spam" else "safe"
        icon = "⚠️" if label == "Spam" else "✅"
        st.markdown(f'<div class="result {css}">{icon} {label}<br><br>Confidence: {conf:.2%}</div>', unsafe_allow_html=True)
        st.progress(conf)
    else:
        st.warning("Please enter a message.")

st.markdown('<div class="footer">Streamlit • Amirthavanan</div>', unsafe_allow_html=True)

# Save this as LOGR.py and run:
# !streamlit run LOGR.py
