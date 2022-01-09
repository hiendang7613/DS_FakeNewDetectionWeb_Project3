import streamlit as st
import pickle
import re
import pandas as pd
from vncorenlp import VnCoreNLP
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')

def dummy(x): return x
class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self):
      self.pattern = "[^a-z0-9A-Z_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ ]*"
      with open("vietnamese-stopwords.txt",encoding="utf-8") as f :
        self.stop_word = f.read().splitlines()
      return

    def fit(self, X_df, y=None):
      return self
      
    def transform(self, X_df, y=None):
      data = []
      for text in X_df['text']:
        # Chuẩn hoá cột text
        # 1.Loại bỏ ký tự đặc biệt
        text = re.sub(self.pattern, '',text)
        # 2.Tách từ
        text = annotator.tokenize(text)
        # 3.Loại bỏ stop_word
        text = [word for word in text[0] if word not in self.stop_word]
        # 4.Chuẩn hoá chữ thường
        text = [word.lower() for word in text]

        data.append(text)
      return pd.DataFrame({'text':data})

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



annotator = VnCoreNLP("VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx2g')

#== Load Model == 
SVCModel = pickle.load(open("model/svc_.model", 'rb'))
LGRModel = pickle.load(open("model/lgr_.model", 'rb'))
PACModel = pickle.load(open("model/pac_.model", 'rb'))
RFCModel = pickle.load(open("model/rfc_.model", 'rb'))
GBCModel = pickle.load(open("model/gbc_.model", 'rb'))
MNBModel = pickle.load(open("model/mnb_.model", 'rb'))

def main():
    # define model
    st.header("Fake News Detection Web")

    st.subheader("1. Nhập đoạn văn bản bạn muốn kiểm tra:")
    content = pd.DataFrame({'text': [''], 'domain': ['']})
    content.text = st.text_input("Nhập vào đoạn văn bản")

    st.subheader("2. Chọn mô hình bạn mong muốn:")
    df = pd.DataFrame({
        'modelName': [
            "Support Vector Classification",
            "Logistic Regression", 
            "Passive Aggressive Classifier", 
            "Random Forest Classifier", 
            "Gradient Boosting Classifier", 
            "Multinomial Naive Bayes"],
        'value': [1, 2, 3, 4, 5, 6]
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
                if(model == 'Support Vector Classification'):
                    rs = SVCModel.predict(content)
                if(model == 'Logistic Regression'):
                    rs = LGRModel.predict(content)
                if(model == 'Passive Aggressive Classifier'):
                    rs = PACModel.predict(content)
                if(model == 'Random Forest Classifier'):
                    rs = RFCModel.predict(content)
                if(model == 'Gradient Boosting Classifier'):
                    rs = GBCModel.predict(content)
                if(model == 'Multinomial Naive Bayes'):
                    rs = MNBModel.predict(content)
                st.subheader("4. Kết quả:")
                if(rs[0] == 1):
                    st.write("Tin giả")
                else:
                    st.write("Tin thật")

if __name__ == "__main__":
    main()






