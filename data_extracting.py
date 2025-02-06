import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extracting(data_clean, data_real):
    data_clean = pd.DataFrame(data_clean)
    data_real = pd.DataFrame(data_real)

    A = data_clean['text_tokens_stemmed']
    B = data_clean['Label']
    C = data_real['text_tokens_stemmed']

    data_clean = data_clean.drop(columns=['score', 'text_clean', 'text_Stopword', 'text_tokens'])

    A = A.fillna('')
    C = C.fillna('')

    tfid_vectorizer = TfidfVectorizer()

    A_fit_tfid = tfid_vectorizer.fit_transform(A)
    A_tfid = tfid_vectorizer.transform(A)
    C_tfid = tfid_vectorizer.transform(C)

    A_tfid.toarray()
    C_tfid.toarray()

    return A_tfid, B, C_tfid, A_fit_tfid