from flask import Flask, render_template, request
from google_play_scraper import app
from wordcloud import WordCloud
from scrapp import scrapp_3000_data, scrapp_all_data
from data_preprocessing import preprocessing, preprocessing_all
from data_labelling import labelling
from data_extracting import extracting
from naive_bayes import naive_bayes
from svm import svm_classifier
from wordcloudnya import semua, positif
import matplotlib.pyplot as plt
import io
import base64
import re

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    reviews_data = []

    if request.method == 'POST':
        # Ambil URL yang dimasukkan pengguna
        url = request.form['url']
        pattern = r'id=([a-zA-Z0-9\.\_]+)'
        match = re.search(pattern, url)
        if match:
            app_id = match.group(1)

        data_3000 = scrapp_3000_data(app_id)
        data_all = scrapp_all_data(app_id)

        hasil_preprocessing_data_selected = preprocessing(data_3000)
        hasil_labelling_data_selected = labelling(hasil_preprocessing_data_selected)

        hasil_preprocessing_data_all = preprocessing_all(data_all)
        hasil_labelling_data_all = labelling(hasil_preprocessing_data_all)

        A_tfid, B, C_tfid, A_fit_tfid = extracting(hasil_labelling_data_selected, hasil_labelling_data_all)
        overall_accuracy_nb, cr_nb, cm_nb, data_clean_nb = naive_bayes(A_tfid, B, C_tfid, A_fit_tfid, hasil_labelling_data_selected, hasil_labelling_data_all)

        # svm = svm_classifier(A_tfid, B, C_tfid, A_fit_tfid, hasil_labelling_data_selected, hasil_labelling_data_all)

        wordcloud_semua_nb = semua(data_clean_nb)

        # Convert the word cloud to an image
        wordcloud_image = wordcloud_semua_nb.to_array()
        img = io.BytesIO()
        plt.imshow(wordcloud_image, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img, format='PNG')
        img.seek(0)
        plotcloud = base64.b64encode(img.getvalue()).decode()

        reviews_data.append({
            "review": 'cobaa',
            "overall_accuracy_nb": overall_accuracy_nb,
            "cr_nb": cr_nb,
            "cm_nb": cm_nb,
            "plotcloud": plotcloud
        })
        # reviews_data.append({"review": review['content'], "sentiment": sentiment})

    return render_template('index.html', reviews_data=reviews_data)

if __name__ == '__main__':
    app.run(debug=True)
