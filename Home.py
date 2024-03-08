import streamlit as st

st.page_link("Home.py", label="Home", icon="🏠")
st.page_link("pages/DTree.py", label="Page 1", icon="1️⃣")
st.page_link("pages/NaiveBaye.py", label="Page 2", icon="2️⃣", disabled=True)
st.page_link("http://www.google.com", label="Google", icon="🌎")