import matplotlib.pyplot as plt  # Mengimpor pustaka matplotlib untuk membuat grafik
import pandas as pd  # Mengimpor pustaka pandas untuk manipulasi data

def grafik_nb(dr):  # Mendefinisikan fungsi untuk membuat grafik Naive Bayes
    dr['date'] = pd.to_datetime(dr['at'])  # Mengonversi kolom 'at' menjadi tipe datetime
    dr = dr.sort_values(by='date')  # Mengurutkan data berdasarkan kolom 'date'

    # Tambahkan kolom bulan (format YYYY-MM)
    dr['month'] = dr['date'].dt.to_period('M')  # Menambahkan kolom 'month' dengan format periode bulanan

    # Agregasi jumlah sentimen per bulan
    sentiment_per_month = dr.groupby(['month', 'Label NB']).size().unstack(fill_value=0)
    # Mengelompokkan data berdasarkan bulan dan label sentimen Naive Bayes, lalu menghitung jumlahnya

    # Plot Line Chart
    sentiment_per_month.plot(kind='line', figsize=(10, 6))  # Membuat grafik garis dengan ukuran 10x6

    # Tambahkan judul dan label sumbu
    plt.title('Tren Sentimen Analisis dengan Menggunakan Algoritma Naive Bayes per Bulan di Indonesia')
    # Menambahkan judul grafik
    plt.xlabel('Bulan')  # Menambahkan label pada sumbu X
    plt.ylabel('Jumlah Sentimen')  # Menambahkan label pada sumbu Y
    plt.xticks(rotation=45)  # Memutar label sumbu X sebesar 45 derajat
    plt.legend(title='Sentimen', loc='upper left')  # Menambahkan legenda dengan judul 'Sentimen'
    plt.tight_layout()  # Menyesuaikan tata letak agar tidak ada elemen yang tumpang tindih

    # Simpan plot sebagai file gambar
    plt.savefig('static/grafik_nb.png', format='png')  # Menyimpan grafik sebagai file PNG

def grafik_svm_linear(dr):  # Mendefinisikan fungsi untuk membuat grafik SVM Linear
    dr['date'] = pd.to_datetime(dr['at'])  # Mengonversi kolom 'at' menjadi tipe datetime
    dr = dr.sort_values(by='date')  # Mengurutkan data berdasarkan kolom 'date'

    # Tambahkan kolom bulan (format YYYY-MM)
    dr['month'] = dr['date'].dt.to_period('M')  # Menambahkan kolom 'month' dengan format periode bulanan

    # Agregasi jumlah sentimen per bulan
    sentiment_per_month = dr.groupby(['month', 'Label SVM']).size().unstack(fill_value=0)
    # Mengelompokkan data berdasarkan bulan dan label sentimen SVM Linear, lalu menghitung jumlahnya

    # Plot Line Chart
    sentiment_per_month.plot(kind='line', figsize=(10, 6))  # Membuat grafik garis dengan ukuran 10x6

    # Tambahkan judul dan label sumbu
    plt.title('Tren Sentimen Analisis Algoritma SVM Linear per Bulan di Indonesia')
    # Menambahkan judul grafik
    plt.xlabel('Bulan')  # Menambahkan label pada sumbu X
    plt.ylabel('Jumlah Sentimen')  # Menambahkan label pada sumbu Y
    plt.xticks(rotation=45)  # Memutar label sumbu X sebesar 45 derajat
    plt.legend(title='Sentimen', loc='upper left')  # Menambahkan legenda dengan judul 'Sentimen'
    plt.tight_layout()  # Menyesuaikan tata letak agar tidak ada elemen yang tumpang tindih

    # Simpan plot sebagai file gambar
    plt.savefig('static/grafik_svm_linear.png', format='png')  # Menyimpan grafik sebagai file PNG

def grafik_svm_nonlinear(dr):  # Mendefinisikan fungsi untuk membuat grafik SVM Non Linear
    dr['date'] = pd.to_datetime(dr['at'])  # Mengonversi kolom 'at' menjadi tipe datetime
    dr = dr.sort_values(by='date')  # Mengurutkan data berdasarkan kolom 'date'

    # Tambahkan kolom bulan (format YYYY-MM)
    dr['month'] = dr['date'].dt.to_period('M')  # Menambahkan kolom 'month' dengan format periode bulanan

    # Agregasi jumlah sentimen per bulan
    sentiment_per_month = dr.groupby(['month', 'Label SVM']).size().unstack(fill_value=0)
    # Mengelompokkan data berdasarkan bulan dan label sentimen SVM Non Linear, lalu menghitung jumlahnya

    # Plot Line Chart
    sentiment_per_month.plot(kind='line', figsize=(10, 6))  # Membuat grafik garis dengan ukuran 10x6

    # Tambahkan judul dan label sumbu
    plt.title('Tren Sentimen Analisis Algoritma SVM Non Linear per Bulan di Indonesia')
    # Menambahkan judul grafik
    plt.xlabel('Bulan')  # Menambahkan label pada sumbu X
    plt.ylabel('Jumlah Sentimen')  # Menambahkan label pada sumbu Y
    plt.xticks(rotation=45)  # Memutar label sumbu X sebesar 45 derajat
    plt.legend(title='Sentimen', loc='upper left')  # Menambahkan legenda dengan judul 'Sentimen'
    plt.tight_layout()  # Menyesuaikan tata letak agar tidak ada elemen yang tumpang tindih

    # Simpan plot sebagai file gambar
    plt.savefig('static/grafik_svm_nonlinear.png', format='png')  # Menyimpan grafik sebagai file PNG