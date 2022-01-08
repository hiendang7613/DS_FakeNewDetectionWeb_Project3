
# from do_an_3 import dummy
# from do_an_3 import MergeCol
# from do_an_3 import PreprocessDomain
# from do_an_3 import PreprocessText
import streamlit as st
import pandas as pd
import pickle

import warnings
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline, make_pipeline
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
import re
from vncorenlp import VnCoreNLP
import pandas as pd
warnings.filterwarnings('ignore')

annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar",
                      annotators="wseg,pos,ner,parse", max_heap_size='-Xmx2g')

# load model


def dummy(x): return x


class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pattern = "[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ ]*"
        with open("vietnamese-stopwords.txt", encoding="utf-8") as f:
            self.stop_word = f.read().splitlines()
        return

    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df, y=None):
        data = []
        for text in X_df['text']:
            # Chuẩn hoá cột text
            # 1.Loại bỏ ký tự đặc biệt
            text = re.sub(self.pattern, '', text)
            # 2.Tách từ
            text = annotator.tokenize(text)
            # 3.Loại bỏ stop_word
            text = [word for word in text[0] if word not in self.stop_word]
            # 4.Chuẩn hoá chữ thường
            text = [word.lower() for word in text]

            data.append(text)
        return pd.DataFrame({'text': data})


class PreprocessDomain(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.secret = '$'
        return

    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df, y=None):
        data = []
        for domain in X_df['domain']:
            # Chuẩn hoá cột domain
            # 1.Loại bỏ ký tự đặc biệt
            domain = domain.replace('.', ' ')
            # 2.Tách từ
            domain = annotator.tokenize(domain)
            # 3.Chuẩn hoá chữ thường
            domain = [word.lower() for word in domain[0]]
            # 4.Chuẩn hoá chữ thường
            domain = [word + self.secret for word in domain]
            data.append(domain)
        return pd.DataFrame({'domain': data})


class MergeCol(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X_df, y=None):
        return self

    def transform(self, X_df, y=None):
        data = []
        for x in X_df:
            data.append(x[0]+x[1])
        return data


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
                    st.write("m1")
                    rs = LGRModel.predict(content)
                if(model == 'RandomForestClassifier'):
                    st.write("m2")

                    rs = RFCModel.predict(content)
                if(model == 'MultinomialNB'):
                    st.write("m3")

                    rs = MNBModel.predict(content)
                if(model == 'SVC'):
                    st.write("m4")

                    rs = SVCModel.predict(content)
                st.subheader("4. Kết quả:")
                if(rs[0] == 1):

                    st.write("Tin giả")
                else:
                    st.write("Tin thật")


if __name__ == "__main__":
    main()
