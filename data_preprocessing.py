import re  # Mengimpor modul untuk operasi regex
import nltk  # Mengimpor modul untuk Natural Language Processing
import os  # Mengimpor modul untuk operasi sistem file
import pandas as pd  # Mengimpor modul untuk manipulasi data dengan DataFrame

def preprocessing(datanya):  # Fungsi untuk preprocessing data tertentu
    file_path = 'sample_data/the_data_hasiltextpreprocessing.csv'  # Lokasi file hasil preprocessing
    if os.path.exists(file_path):  # Mengecek apakah file hasil preprocessing sudah ada
        my_df = pd.read_csv(file_path, sep=",")  # Membaca file CSV jika sudah ada
    else:  # Jika file belum ada
        my_df = datanya[['userName', 'score', 'at', 'content']].sort_values(by=['at'], ascending=False)  # Mengurutkan data berdasarkan kolom 'at'
        my_df = my_df.dropna(subset=['content'])  # Menghapus baris dengan nilai kosong pada kolom 'content'

        print('casefolding   data')  # Menampilkan pesan proses casefolding
        my_df = casefolding(my_df)  # Memanggil fungsi casefolding
        print('filtering selected data')  # Menampilkan pesan proses filtering
        my_df = filtering(my_df)  # Memanggil fungsi filtering
        print('tokenizing selected data')  # Menampilkan pesan proses tokenizing
        my_df = tokenizing(my_df)  # Memanggil fungsi tokenizing
        print('stemming selected data')  # Menampilkan pesan proses stemming
        my_df = stemming(my_df)  # Memanggil fungsi stemming
        my_df.to_csv(file_path, index=False)  # Menyimpan hasil preprocessing ke file CSV

    return my_df  # Mengembalikan DataFrame hasil preprocessing

def preprocessing_all(datanya):  # Fungsi untuk preprocessing semua data
    file_path = 'sample_data/the_data_all_hasiltextpreprocessing.csv'  # Lokasi file hasil preprocessing semua data
    if os.path.exists(file_path):  # Mengecek apakah file hasil preprocessing sudah ada
        my_df = pd.read_csv(file_path, sep=",")  # Membaca file CSV jika sudah ada
    else:  # Jika file belum ada
        my_df = datanya[['userName', 'score', 'at', 'content']].sort_values(by=['at'], ascending=False)  # Mengurutkan data berdasarkan kolom 'at'
        my_df = my_df.dropna(subset=['content'])  # Menghapus baris dengan nilai kosong pada kolom 'content'

        print('casefolding all data')  # Menampilkan pesan proses casefolding
        my_df = casefolding(my_df)  # Memanggil fungsi casefolding
        print('filtering all data')  # Menampilkan pesan proses filtering
        my_df = filtering(my_df)  # Memanggil fungsi filtering
        print('tokenizing all data')  # Menampilkan pesan proses tokenizing
        my_df = tokenizing(my_df)  # Memanggil fungsi tokenizing
        print('stemming all data')  # Menampilkan pesan proses stemming
        my_df = stemming(my_df)  # Memanggil fungsi stemming
        my_df.to_csv(file_path, index=False)  # Menyimpan hasil preprocessing ke file CSV

    return my_df  # Mengembalikan DataFrame hasil preprocessing

def casefolding(my_df):  # Fungsi untuk casefolding (mengubah teks menjadi huruf kecil)
    def clean_text(df, text_field, new_text_field_name):  # Fungsi untuk membersihkan teks
        my_df[new_text_field_name] = my_df[text_field].str.lower()  # Mengubah teks menjadi huruf kecil
        my_df[new_text_field_name] = my_df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  # Menghapus simbol, URL, dan mention
        my_df[new_text_field_name] = my_df[new_text_field_name].str.replace(r"\d+", "")  # Menghapus angka
        return my_df  # Mengembalikan DataFrame yang telah dibersihkan
    my_df['text_clean'] = my_df['content'].str.lower()  # Menambahkan kolom 'text_clean' dengan teks huruf kecil
    data_clean = clean_text(my_df, 'content', 'text_clean')  # Memanggil fungsi clean_text

    return data_clean  # Mengembalikan DataFrame yang telah dibersihkan

def filtering(data_clean):  # Fungsi untuk filtering (menghapus stopword)
    nltk.download('stopwords')  # Mengunduh daftar stopword
    from nltk.corpus import stopwords  # Mengimpor daftar stopword

    stop = stopwords.words('indonesian')  # Mengambil daftar stopword bahasa Indonesia
    data_clean['text_Stopword'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))  # Menghapus stopword dari teks

    return data_clean  # Mengembalikan DataFrame yang telah difilter

def tokenizing(data_clean):  # Fungsi untuk tokenizing (memecah teks menjadi token)
    nltk.download('punkt')  # Mengunduh tokenizer
    nltk.download('punkt_tab')  # Mengunduh tokenizer tambahan
    from nltk.tokenize import word_tokenize, sent_tokenize  # Mengimpor fungsi tokenizing

    data_clean['text_tokens'] = data_clean['text_Stopword'].apply(lambda x: word_tokenize(x))  # Memecah teks menjadi token

    return data_clean  # Mengembalikan DataFrame yang telah ditokenisasi

def stemming(data_clean):  # Fungsi untuk stemming (mengubah kata menjadi bentuk dasar)
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Mengimpor library stemming bahasa Indonesia
    factory = StemmerFactory()  # Membuat objek StemmerFactory
    stemmer = factory.create_stemmer()  # Membuat objek stemmer

    def stemmed_wrapper(term):  # Fungsi untuk stemming kata
        return stemmer.stem(term)  # Mengembalikan kata dalam bentuk dasar
    term_dict = {}  # Dictionary untuk menyimpan kata dan hasil stemming
    hitung = 0  # Variabel untuk menghitung jumlah kata

    for document in data_clean['text_tokens']:  # Iterasi setiap dokumen
        for term in document:  # Iterasi setiap kata dalam dokumen
            if term not in term_dict:  # Jika kata belum ada di dictionary
                term_dict[term] = ' '  # Tambahkan kata ke dictionary

    for term in term_dict:  # Iterasi setiap kata dalam dictionary
        term_dict[term] = stemmed_wrapper(term)  # Stemming kata
        hitung += 1  # Menambah jumlah kata yang diproses

    def get_stemmed_term(document):  # Fungsi untuk mendapatkan kata yang telah distemming
        return [term_dict[term] for term in document]  # Mengembalikan kata yang telah distemming

    data_clean['text_tokens_stemmed'] = data_clean['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x)))  # Menambahkan kolom dengan hasil stemming

    return data_clean  # Mengembalikan DataFrame yang telah distemming
