
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

# load model
from do_an_3 import PreprocessText
from do_an_3 import PreprocessDomain
from do_an_3 import MergeCol
from do_an_3 import dummy

models = {
    " RandomForestClassifier": "models/pac.pkl",
    "LogisticRegression": "models/lr.pkl",
    "MultinomialNB": "models/lr.pkl",
    "SVC": "models/lr.pkl",
}


RFCModel = pickle.load(open("model/rfc_.model", 'rb'))
LGRModel = pickle.load(open("model/lgr_.model", 'rb'))
MNBModel = pickle.load(open("model/mnb_.model", 'rb'))
SVCModel = pickle.load(open("model/svc_.model", 'rb'))


def main():
    # define model

    st.header("Fake News Detection Web")
    st.subheader("1. Nhập đoạn văn bản bạn muốn kiểm tra:")
    content = pd.DataFrame({'text': [''], 'domain': ['']})
    content.text = st.text_input("Nhập vào đoạn văn bản")
    st.subheader("2. Chọn mô hình bạn mong muốn:")
    df = pd.DataFrame({
        'modelName': ["LogisticRegression", "RandomForestClassifier", "MultinomialNB", "SVC"],
        'value': [1, 2, 3, 4]
    })
    rs = []
    model = st.selectbox(
        'Chọn mô hình',
        df['modelName'])
    st.subheader("3. Nhập nguồn bài báo")
    content.domain = st.text_input("Nhập nguồn tin, nếu không có thì bỏ trống")
    if(content.domain[0] == ''):
        content.domain = 'a'
    button = st.button("Kiểm tra")
    thongBao = st.empty()
    if button:
        with st.spinner("Đang kiểm tra ..."):
            if not len(content):
                thongBao.markdown("In put at least a piece of news")
            else:
                if(model == 'LogisticRegression'):
                    rs = LGRModel.predict(content)
                if(model == 'RandomForestClassifier'):
                    rs = RFCModel.predict(content)
                if(model == 'MultinomialNB'):
                    rs = MNBModel.predict(content)
                if(model == 'SVC'):
                    rs = SVCModel.predict(content)
                st.subheader("4. Kết quả:")
                if(rs[0] == 1):

                    st.write("Tin giả")
                else:
                    st.write("Tin thật")


if __name__ == "__main__":
    main()
