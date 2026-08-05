import streamlit as st
import joblib
import sklearn
import numpy
import pandas

st.title("Environment Test")

st.write("joblib imported successfully")
st.write("scikit-learn version:", sklearn.__version__)
st.write("numpy version:", numpy.__version__)
st.write("pandas version:", pandas.__version__)
