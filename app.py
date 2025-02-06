from flask import Flask, render_template, request
from google_play_scraper import app
from wordcloud import WordCloud
from cek import coba
import joblib
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)

# Load model Naive Bayes atau SVM
model = joblib.load('models/naive_bayes.pkl')  # Atau svm.pkl jika menggunakan SVM

@app.route('/', methods=['GET', 'POST'])
def index():
    reviews_data = []

    if request.method == 'POST':
        # Ambil URL yang dimasukkan pengguna
        url = request.form['url']

        # # Ambil ulasan dari Google Play Store
        # reviews = get_reviews(url)

        # # Lakukan analisis sentimen untuk setiap ulasan
        # for review in reviews:
        #     sentiment = predict_sentiment(review['content'])  # Prediksi sentimen
        cobaa = coba(url)
        reviews_data.append({"review": cobaa})
        # reviews_data.append({"review": review['content'], "sentiment": sentiment})

        # # Membuat wordcloud dari ulasan
        # wordcloud_image = generate_wordcloud(reviews_data)

        # return render_template('index.html', reviews_data=reviews_data, wordcloud_image=wordcloud_image)

    return render_template('index.html', reviews_data=reviews_data)

# # Fungsi untuk mengambil ulasan dari Google Play Store
# def get_reviews(url):
#     # Ambil ulasan menggunakan Google Play Scraper
#     result = app.reviews(url, lang='id', country='id')
#     return result[0]  # Ambil daftar ulasan

# # Fungsi untuk melakukan prediksi sentimen (positif/negatif)
# def predict_sentiment(text):
#     return model.predict([text])[0]

# # Fungsi untuk membuat wordcloud dari teks ulasan
# def generate_wordcloud(reviews_data):
#     text = " ".join([review['review'] for review in reviews_data])
#     wordcloud = WordCloud(width=800, height=400).generate(text)

#     # Menyimpan wordcloud ke dalam buffer image
#     img = io.BytesIO()
#     wordcloud.to_image().save(img, format='PNG')
#     img.seek(0)
#     img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

#     return img_base64

if __name__ == '__main__':
    app.run(debug=True)
