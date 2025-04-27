from wordcloud import WordCloud, STOPWORDS  # Import library untuk membuat wordcloud dan stopwords
import matplotlib.pyplot as plt  # Import library untuk plotting
from io import BytesIO  # Import library untuk manipulasi data dalam memori

def wordcloud_semua_nb(data):  # Definisi fungsi untuk membuat wordcloud dari semua data dengan Naive Bayes
    def plot_cloud(wordcloud):  # Definisi fungsi untuk plotting wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran tertentu
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menghilangkan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_semua_nb.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file

    all_words = ' '.join([tweets for tweets in data['text_tokens_stemmed'].fillna('')])  # Menggabungkan semua teks yang sudah di-stem

    wordcloud = WordCloud(  # Membuat objek WordCloud
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Seed untuk randomisasi
        background_color='black',  # Warna latar belakang
        colormap='Blues_r',  # Skema warna
        collocations=False,  # Tidak menggabungkan kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan stopwords bawaan
    ).generate(all_words)  # Membuat wordcloud dari teks

    plot_cloud(wordcloud)  # Memanggil fungsi untuk plotting wordcloud

def wordcloud_positif_nb(data):  # Fungsi untuk membuat wordcloud dari data positif dengan Naive Bayes
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran tertentu
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menghilangkan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_positif_nb.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file

    positif_tweets = data[data['Label NB'] == 'positif'].fillna('')  # Memfilter data dengan label positif dan mengisi nilai kosong

    positif_words = ' '.join([tweets for tweets in positif_tweets['text_tokens_stemmed']])  # Menggabungkan teks dari data positif

    wordcloud = WordCloud(  # Membuat objek WordCloud
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Seed untuk randomisasi
        background_color='black',  # Warna latar belakang
        colormap='Blues_r',  # Skema warna
        collocations=False,  # Tidak menggabungkan kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan stopwords bawaan
    ).generate(positif_words)  # Membuat wordcloud dari teks positif

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_negatif_nb(data):  # Fungsi untuk membuat wordcloud dari data negatif dengan Naive Bayes
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran tertentu
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menghilangkan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_negatif_nb.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file

    negatif_tweets = data[data['Label NB'] == 'negatif'].fillna('')  # Memfilter data dengan label negatif dan mengisi nilai kosong

    negatif_words = ' '.join([tweets for tweets in negatif_tweets['text_tokens_stemmed']])  # Menggabungkan teks dari data negatif

    wordcloud = WordCloud(  # Membuat objek WordCloud
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Seed untuk randomisasi
        background_color='black',  # Warna latar belakang
        colormap='Blues_r',  # Skema warna
        collocations=False,  # Tidak menggabungkan kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan stopwords bawaan
    ).generate(negatif_words)  # Membuat wordcloud dari teks negatif

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_semua_svm_nonlinear(data):  # Fungsi untuk membuat wordcloud dari semua data dengan SVM Nonlinear
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran tertentu
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menghilangkan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_semua_svm_nonlinear.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file

    all_words = ' '.join([tweets for tweets in data['text_tokens_stemmed'].fillna('')])  # Menggabungkan semua teks yang sudah di-stem dari data

    wordcloud = WordCloud(  # Membuat objek WordCloud
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Seed untuk randomisasi
        background_color='black',  # Warna latar belakang
        colormap='Blues_r',  # Skema warna
        collocations=False,  # Tidak menggabungkan kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan stopwords bawaan
    ).generate(all_words)  # Membuat wordcloud dari teks

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_positif_svm_nonlinear(data):  # Fungsi untuk membuat wordcloud dari data positif dengan SVM Nonlinear
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran tertentu
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menghilangkan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_positif_svm_nonlinear.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file

    positif_tweets = data[data['Label SVM'] == 'positif'].fillna('')  # Memfilter data dengan label positif dan mengisi nilai kosong

    positif_words = ' '.join([tweets for tweets in positif_tweets['text_tokens_stemmed']])  # Menggabungkan teks dari data positif

    wordcloud = WordCloud(  # Membuat objek WordCloud
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Seed untuk randomisasi
        background_color='black',  # Warna latar belakang
        colormap='Blues_r',  # Skema warna
        collocations=False,  # Tidak menggabungkan kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan stopwords bawaan
    ).generate(positif_words)  # Membuat wordcloud dari teks positif

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_negatif_svm_nonlinear(data):  # Fungsi untuk membuat wordcloud dari data negatif dengan SVM Nonlinear
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran tertentu
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menghilangkan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_negatif_svm_nonlinear.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file

    negatif_tweets = data[data['Label SVM'] == 'negatif'].fillna('')  # Memfilter data dengan label negatif dan mengisi nilai kosong

    negatif_words = ' '.join([tweets for tweets in negatif_tweets['text_tokens_stemmed']])  # Menggabungkan teks dari data negatif

    wordcloud = WordCloud(  # Membuat objek WordCloud
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Seed untuk randomisasi
        background_color='black',  # Warna latar belakang
        colormap='Blues_r',  # Skema warna
        collocations=False,  # Tidak menggabungkan kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan stopwords bawaan
    ).generate(negatif_words)  # Membuat wordcloud dari teks negatif

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_semua_svm_linear(data):  # Fungsi untuk membuat dan menyimpan wordcloud dari data
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran 10x8 dan layout yang rapat
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menonaktifkan tampilan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_semua_svm_linear.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file PNG dengan pengaturan tanpa margin

    all_words = ' '.join([tweets for tweets in data['text_tokens_stemmed'].fillna('')])  # Menggabungkan semua teks dari kolom 'text_tokens_stemmed' menjadi satu string, mengganti nilai NaN dengan string kosong

    wordcloud = WordCloud(  # Membuat objek WordCloud dengan pengaturan tertentu
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Menentukan seed untuk hasil acak yang konsisten
        background_color='black',  # Warna latar belakang wordcloud
        colormap='Blues_r',  # Skema warna yang digunakan (biru terbalik)
        collocations=False,  # Menonaktifkan penggabungan kata-kata umum
        stopwords=STOPWORDS  # Daftar kata-kata yang akan diabaikan
    ).generate(all_words)  # Membuat wordcloud berdasarkan teks yang digabungkan

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_positif_svm_linear(data):  # Fungsi untuk membuat wordcloud dari data positif dengan SVM Linear
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran 10x8 dan layout yang rapat
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menonaktifkan tampilan sumbu
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_positif_svm_linear.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file PNG dengan pengaturan tanpa margin

    positif_tweets = data[data['Label SVM'] == 'positif'].fillna('')  # Memfilter data dengan label positif dan mengganti nilai kosong dengan string kosong

    positif_words = ' '.join([tweets for tweets in positif_tweets['text_tokens_stemmed']])  # Menggabungkan teks dari data positif menjadi satu string

    wordcloud = WordCloud(  # Membuat objek WordCloud dengan pengaturan tertentu
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Menentukan seed untuk hasil acak yang konsisten
        background_color='black',  # Warna latar belakang wordcloud
        colormap='Blues_r',  # Skema warna yang digunakan (biru terbalik)
        collocations=False,  # Menonaktifkan penggabungan kata-kata umum
        stopwords=STOPWORDS  # Daftar kata-kata yang akan diabaikan
    ).generate(positif_words)  # Membuat wordcloud berdasarkan teks yang digabungkan

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud

def wordcloud_negatif_svm_linear(data):  # Fungsi untuk membuat wordcloud dari data dengan label SVM negatif
    def plot_cloud(wordcloud):  # Fungsi untuk menampilkan dan menyimpan wordcloud
        plt.figure(figsize=(10, 8), tight_layout=True)  # Membuat figure dengan ukuran 10x8 dan tata letak yang rapat
        plt.imshow(wordcloud, interpolation='bilinear')  # Menampilkan wordcloud dengan interpolasi bilinear
        plt.axis('off')  # Menyembunyikan sumbu (axis) pada plot
        plt.margins(0, 0)  # Mengatur margin menjadi nol
        plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu x
        plt.gca().yaxis.set_major_locator(plt.NullLocator())  # Menghilangkan locator sumbu y
        plt.savefig('static/wordcloud_negatif_svm_linear.png', format='png', bbox_inches='tight', pad_inches=0)  # Menyimpan wordcloud ke file PNG dengan nama tertentu

    negatif_tweets = data[data['Label SVM'] == 'negatif'].fillna('')  # Memfilter data untuk hanya mengambil tweet dengan label SVM negatif, mengganti nilai NaN dengan string kosong

    negatif_words = ' '.join([tweets for tweets in negatif_tweets['text_tokens_stemmed']])  # Menggabungkan semua teks dari kolom 'text_tokens_stemmed' menjadi satu string besar

    wordcloud = WordCloud(  # Membuat objek WordCloud dengan parameter tertentu
        width=3000,  # Lebar wordcloud
        height=2000,  # Tinggi wordcloud
        random_state=3,  # Menentukan seed untuk hasil yang konsisten
        background_color='black',  # Warna latar belakang wordcloud
        colormap='Blues_r',  # Skema warna untuk wordcloud
        collocations=False,  # Tidak menggabungkan kata-kata yang sering muncul bersama
        stopwords=STOPWORDS  # Menggunakan daftar kata-kata umum yang akan diabaikan
    ).generate(negatif_words)  # Membuat wordcloud berdasarkan teks negatif

    plot_cloud(wordcloud)  # Memanggil fungsi untuk menampilkan dan menyimpan wordcloud