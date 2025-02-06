from flask import Flask, render_template, request
from google_play_scraper import app
from wordcloud import WordCloud
from cek import coba
import joblib
import matplotlib.pyplot as plt
import io
import base64
import re

app = Flask(__name__)

# Load model Naive Bayes atau SVM
model = joblib.load('models/naive_bayes.pkl')  # Atau svm.pkl jika menggunakan SVM

@app.route('/', methods=['GET', 'POST'])
def index():
    reviews_data = []

    if request.method == 'POST':
        # Ambil URL yang dimasukkan pengguna
        url = request.form['url']

        # Regex untuk mengekstrak ID aplikasi
        pattern = r'id=([a-zA-Z0-9\.\_]+)'

        # Mencocokkan pola dan mengambil bagian ID aplikasi
        match = re.search(pattern, url)

        if match:
            app_id = match.group(1)  # Mengambil bagian ID aplikasi (com.jobstreet.jobstreet)

        cobaa = coba(app_id)
        reviews_data.append({"review": cobaa})
        # reviews_data.append({"review": review['content'], "sentiment": sentiment})

    return render_template('index.html', reviews_data=reviews_data)

if __name__ == '__main__':
    app.run(debug=True)
