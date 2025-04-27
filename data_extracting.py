import pandas as pd  # Mengimpor library pandas untuk manipulasi data
from sklearn.feature_extraction.text import TfidfVectorizer  # Mengimpor TfidfVectorizer untuk ekstraksi fitur teks

def extracting(data_clean, data_real):  # Mendefinisikan fungsi `extracting` dengan dua parameter: `data_clean` dan `data_real`
    data_clean = pd.DataFrame(data_clean)  # Mengonversi `data_clean` menjadi DataFrame pandas
    data_real = pd.DataFrame(data_real)  # Mengonversi `data_real` menjadi DataFrame pandas

    A = data_clean['text_tokens_stemmed']  # Mengambil kolom `text_tokens_stemmed` dari `data_clean` dan menyimpannya di variabel `A`
    B = data_clean['Label']  # Mengambil kolom `Label` dari `data_clean` dan menyimpannya di variabel `B`
    C = data_real['text_tokens_stemmed']  # Mengambil kolom `text_tokens_stemmed` dari `data_real` dan menyimpannya di variabel `C`

    data_clean = data_clean.drop(columns=['score', 'text_clean', 'text_Stopword', 'text_tokens'])
    # Menghapus kolom `score`, `text_clean`, `text_Stopword`, dan `text_tokens` dari `data_clean`

    A = A.fillna('')  # Mengisi nilai NaN di kolom `A` dengan string kosong
    C = C.fillna('')  # Mengisi nilai NaN di kolom `C` dengan string kosong

    tfid_vectorizer = TfidfVectorizer()  # Membuat objek TfidfVectorizer untuk mengubah teks menjadi representasi numerik

    A_fit_tfid = tfid_vectorizer.fit_transform(A)  # Melakukan fit dan transformasi pada data `A`
    A_tfid = tfid_vectorizer.transform(A)  # Melakukan transformasi pada data `A` menggunakan model yang sudah di-fit
    C_tfid = tfid_vectorizer.transform(C)  # Melakukan transformasi pada data `C` menggunakan model yang sama

    A_tfid.toarray()  # Mengonversi hasil transformasi `A_tfid` menjadi array (tidak disimpan ke variabel)
    C_tfid.toarray()  # Mengonversi hasil transformasi `C_tfid` menjadi array (tidak disimpan ke variabel)

    return A_tfid, B, C_tfid, A_fit_tfid  # Mengembalikan hasil transformasi `A_tfid`, label `B`, `C_tfid`, dan `A_fit_tfid`

