import streamlit as st
import pandas as pd
import numpy as np
st.header("Fake News Detection Web")
st.subheader("1. Nhập đoạn văn bản bạn muốn kiểm tra:")
text = st.text_input("Nhập vào đoạn văn bản")
st.subheader("2. Chọn mô hình bạn mong muốn:")
df = pd.DataFrame({
    'modelName': ["LogisticRegression", "RandomForestClassifier", "MultinomialNB", "SVC"],
    'value': [1, 2, 3, 4]
})

option = st.selectbox(
    'Chọn mô hình',
    df['modelName'])

st.subheader("3. Kết quả:")
st.write(option)
