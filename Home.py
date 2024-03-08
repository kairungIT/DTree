import streamlit as st

st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/DTree.py", label="DecisionTree", icon="1️⃣")
st.page_link("pages/NaiveBaye.py", label="NaiveBaye", icon="2️⃣", disabled=True)
st.page_link("http://www.google.com", label="Google", icon="🌎")