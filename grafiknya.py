import matplotlib.pyplot as plt
import pandas as pd

def grafik_nb(dr):
    dr['date'] = pd.to_datetime(dr['at'])
    dr = dr.sort_values(by='date')

    # Tambahkan kolom bulan (format YYYY-MM)
    dr['month'] = dr['date'].dt.to_period('M')

    # Agregasi jumlah sentimen per bulan
    sentiment_per_month = dr.groupby(['month', 'Label NB']).size().unstack(fill_value=0)

    # Plot Line Chart
    sentiment_per_month.plot(kind='line', figsize=(10, 6))

    # Tambahkan judul dan label sumbu
    plt.title('Tren Sentimen Analisis dengan Menggunakan Algoritma Naive Bayes per Bulan di Indonesia')
    plt.xlabel('Bulan')
    plt.ylabel('Jumlah Sentimen')
    plt.xticks(rotation=45)
    plt.legend(title='Sentimen', loc='upper left')
    plt.tight_layout()

    # Simpan plot sebagai file gambar
    plt.savefig('static/grafik_nb.png', format='png')

def grafik_svm(dr):
    dr['date'] = pd.to_datetime(dr['at'])
    dr = dr.sort_values(by='date')

    # Tambahkan kolom bulan (format YYYY-MM)
    dr['month'] = dr['date'].dt.to_period('M')

    # Agregasi jumlah sentimen per bulan
    sentiment_per_month = dr.groupby(['month', 'Label SVM']).size().unstack(fill_value=0)

    # Plot Line Chart
    sentiment_per_month.plot(kind='line', figsize=(10, 6))

    # Tambahkan judul dan label sumbu
    plt.title('Tren Sentimen Analisis Algoritma SVM per Bulan di Indonesia')
    plt.xlabel('Bulan')
    plt.ylabel('Jumlah Sentimen')
    plt.xticks(rotation=45)
    plt.legend(title='Sentimen', loc='upper left')
    plt.tight_layout()

    # Tampilkan plot
    plt.show()
    # Simpan plot sebagai file gambar
    plt.savefig('static/grafik_svm.png')