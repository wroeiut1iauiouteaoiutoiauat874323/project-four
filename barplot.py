import matplotlib.pyplot as plt  # Mengimpor modul matplotlib.pyplot untuk membuat plot
import pandas as pd  # Mengimpor modul pandas untuk manipulasi data

def barplot_nb(dr):
    # Mengonversi kolom 'at' menjadi tipe datetime dan mengurutkan data berdasarkan tanggal
    dr['date'] = pd.to_datetime(dr['at'])  # Mengubah kolom 'at' menjadi datetime
    dr = dr.sort_values(by='date')  # Mengurutkan data berdasarkan kolom 'date'

    # Memformat tanggal menjadi 'YYYY-MM-DD' dan mengonversinya kembali ke datetime
    dr['date'] = dr['date'].dt.strftime('%Y-%m-%d')  # Memformat tanggal menjadi string
    dr['date'] = pd.to_datetime(dr['date'])  # Mengonversi kembali string menjadi datetime

    # Menghitung jumlah nilai dan persentase untuk kolom 'Label NB'
    value_counts = dr['Label NB'].value_counts()  # Menghitung jumlah setiap nilai di kolom 'Label NB'
    percentages = value_counts / value_counts.sum() * 100  # Menghitung persentase setiap nilai

    # Membuat diagram batang dengan persentase
    ax = value_counts.plot(kind='bar')  # Membuat diagram batang berdasarkan jumlah nilai
    for i, (count, percentage) in enumerate(zip(value_counts, percentages)):  # Menambahkan label persentase di atas batang
        ax.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom')  # Menampilkan persentase di atas batang

    plt.title('Sentiment Analysis menggunakan Algoritma Naive Bayes')  # Menambahkan judul pada plot
    plt.xlabel('Sentiment')  # Menambahkan label sumbu X
    plt.ylabel('Total')  # Menambahkan label sumbu Y

    # Menyimpan plot sebagai file gambar
    plt.savefig('static/barplot_nb.png', format='png')  # Menyimpan plot dalam format PNG

def barplot_svm_linear(dr):
    # Mengonversi kolom 'at' menjadi tipe datetime dan mengurutkan data berdasarkan tanggal
    dr['date'] = pd.to_datetime(dr['at'])  # Mengubah kolom 'at' menjadi datetime
    dr = dr.sort_values(by='date')  # Mengurutkan data berdasarkan kolom 'date'

    # Memformat tanggal menjadi 'YYYY-MM-DD' dan mengonversinya kembali ke datetime
    dr['date'] = dr['date'].dt.strftime('%Y-%m-%d')  # Memformat tanggal menjadi string
    dr['date'] = pd.to_datetime(dr['date'])  # Mengonversi kembali string menjadi datetime

    # Menghitung jumlah nilai dan persentase untuk kolom 'Label SVM'
    value_counts = dr['Label SVM'].value_counts()  # Menghitung jumlah setiap nilai di kolom 'Label SVM'
    percentages = value_counts / value_counts.sum() * 100  # Menghitung persentase setiap nilai

    # Membersihkan plot sebelumnya
    plt.clf()  # Membersihkan figure saat ini

    # Membuat diagram batang dengan persentase
    ax = value_counts.plot(kind='bar')  # Membuat diagram batang berdasarkan jumlah nilai
    for i, (count, percentage) in enumerate(zip(value_counts, percentages)):  # Menambahkan label persentase di atas batang
        ax.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom')  # Menampilkan persentase di atas batang

    plt.title('Sentiment Analysis menggunakan Algoritma SVM Linear')  # Menambahkan judul pada plot
    plt.xlabel('Sentiment')  # Menambahkan label sumbu X
    plt.ylabel('Total')  # Menambahkan label sumbu Y

    # Menyimpan plot sebagai file gambar
    plt.savefig('static/barplot_svm_linear.png', format='png')  # Menyimpan plot dalam format PNG

def barplot_svm_nonlinear(dr):
    # Mengonversi kolom 'at' menjadi tipe datetime dan mengurutkan data berdasarkan tanggal
    dr['date'] = pd.to_datetime(dr['at'])  # Mengubah kolom 'at' menjadi datetime
    dr = dr.sort_values(by='date')  # Mengurutkan data berdasarkan kolom 'date'

    # Memformat tanggal menjadi 'YYYY-MM-DD' dan mengonversinya kembali ke datetime
    dr['date'] = dr['date'].dt.strftime('%Y-%m-%d')  # Memformat tanggal menjadi string
    dr['date'] = pd.to_datetime(dr['date'])  # Mengonversi kembali string menjadi datetime

    # Menghitung jumlah nilai dan persentase untuk kolom 'Label SVM'
    value_counts = dr['Label SVM'].value_counts()  # Menghitung jumlah setiap nilai di kolom 'Label SVM'
    percentages = value_counts / value_counts.sum() * 100  # Menghitung persentase setiap nilai

    # Membersihkan plot sebelumnya
    plt.clf()  # Membersihkan figure saat ini

    # Membuat diagram batang dengan persentase
    ax = value_counts.plot(kind='bar')  # Membuat diagram batang berdasarkan jumlah nilai
    for i, (count, percentage) in enumerate(zip(value_counts, percentages)):  # Menambahkan label persentase di atas batang
        ax.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom')  # Menampilkan persentase di atas batang

    plt.title('Sentiment Analysis menggunakan Algoritma SVM Non Linear')  # Menambahkan judul pada plot
    plt.xlabel('Sentiment')  # Menambahkan label sumbu X
    plt.ylabel('Total')  # Menambahkan label sumbu Y

    # Menyimpan plot sebagai file gambar
    plt.savefig('static/barplot_svm_nonlinear.png', format='png')  # Menyimpan plot dalam format PNG