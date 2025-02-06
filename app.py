from flask import Flask, render_template, request
from google_play_scraper import app
from wordcloud import WordCloud
from scrapp import scrapp_3000_data, scrapp_all_data
from data_preprocessing import preprocessing, preprocessing_all
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
        pattern = r'id=([a-zA-Z0-9\.\_]+)'
        match = re.search(pattern, url)
        if match:
            app_id = match.group(1)  # Mengambil bagian ID aplikasi (com.jobstreet.jobstreet)

        cobaa = coba(app_id)
        data_3000 = scrapp_3000_data(app_id)
        data_all = scrapp_all_data(app_id)
        hasil_preprocessing_data_selected = preprocessing(data_3000)
        hasil_preprocessing_data_all = preprocessing_all(data_all)

        reviews_data.append({"review": cobaa})
        # reviews_data.append({"review": review['content'], "sentiment": sentiment})

    return render_template('index.html', reviews_data=reviews_data)

if __name__ == '__main__':
    app.run(debug=True)
