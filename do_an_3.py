"""# Đồ án 3 - Fake news detection

Thành viên : 

Đặng Văn Hiển - 18120363

Trà Anh Toàn - 1812662

Lê Thanh Viễn - 18120647

Nguyễn Trần Nhật Minh - 18120208

Nguyễn Vinh Quang - 18120229
"""


"""# Import"""

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



