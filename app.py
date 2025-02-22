from flask import Flask, render_template, request
from google_play_scraper import app
from wordcloud import WordCloud
from scrapp import scrapp_3000_data, scrapp_all_data
from data_preprocessing import preprocessing, preprocessing_all
from data_labelling import labelling_selected, labelling_all
from data_extracting import extracting
from naive_bayes import naive_bayes
from svm_nonlinear import svm_classifier_nonlinear
from svm_linear import svm_classifier_linear
from wordcloudnya import (
    wordcloud_semua_nb,
    wordcloud_positif_nb,
    wordcloud_negatif_nb,
    wordcloud_semua_svm_nonlinear,
    wordcloud_positif_svm_nonlinear,
    wordcloud_negatif_svm_nonlinear,
    wordcloud_semua_svm_linear,
    wordcloud_positif_svm_linear,
    wordcloud_negatif_svm_linear,
)
from grafiknya import grafik_nb, grafik_svm_nonlinear, grafik_svm_linear
from barplot import barplot_nb, barplot_svm_nonlinear, barplot_svm_linear
import matplotlib.pyplot as plt
import io
import base64
import re

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    reviews_data = []

    if request.method == "POST":
        # Ambil URL yang dimasukkan pengguna
        url = request.form["url"]
        pattern = r"id=([a-zA-Z0-9\.\_]+)"
        match = re.search(pattern, url)
        if match:
            app_id = match.group(1)

        data_3000 = scrapp_3000_data(app_id)
        data_all = scrapp_all_data(app_id)

        hasil_preprocessing_data_selected = preprocessing(data_3000)
        hasil_labelling_data_selected = labelling_selected(
            hasil_preprocessing_data_selected
        )

        hasil_preprocessing_data_all = preprocessing_all(data_all)
        hasil_labelling_data_all = labelling_all(hasil_preprocessing_data_all)

        A_tfid, B, C_tfid, A_fit_tfid = extracting(
            hasil_labelling_data_selected, hasil_labelling_data_all
        )
        (
            overall_accuracy_nb,
            cr_nb,
            cm_nb,
            data_clean_nb,
            data_real_nb,
            jumlah_data_clean_nb,
        ) = naive_bayes(
            A_tfid,
            B,
            C_tfid,
            A_fit_tfid,
            hasil_labelling_data_selected,
            hasil_labelling_data_all,
        )

        (
            overall_accuracy_svm_nonlinear,
            cr_svm_nonlinear,
            cm_svm_nonlinear,
            data_clean_svm_nonlinear,
            data_real_svm_nonlinear,
            jumlah_data_clean_svm_nonlinear,
        ) = svm_classifier_nonlinear(
            A_tfid,
            B,
            C_tfid,
            A_fit_tfid,
            hasil_labelling_data_selected,
            hasil_labelling_data_all,
        )

        (
            overall_accuracy_svm_linear,
            cr_svm_linear,
            cm_svm_linear,
            data_clean_svm_linear,
            data_real_svm_linear,
            jumlah_data_clean_svm_linear,
        ) = svm_classifier_linear(
            A_tfid,
            B,
            C_tfid,
            A_fit_tfid,
            hasil_labelling_data_selected,
            hasil_labelling_data_all,
        )

        wordcloud_semua_nb(data_real_nb)
        wordcloud_positif_nb(data_real_nb)
        wordcloud_negatif_nb(data_real_nb)

        wordcloud_semua_svm_nonlinear(data_real_svm_nonlinear)
        wordcloud_positif_svm_nonlinear(data_real_svm_nonlinear)
        wordcloud_negatif_svm_nonlinear(data_real_svm_nonlinear)

        wordcloud_semua_svm_linear(data_real_svm_linear)
        wordcloud_positif_svm_linear(data_real_svm_linear)
        wordcloud_negatif_svm_linear(data_real_svm_linear)

        grafik_nb(data_real_nb)
        grafik_svm_nonlinear(data_real_svm_nonlinear)
        grafik_svm_linear(data_real_svm_linear)

        barplot_nb(data_real_nb)
        barplot_svm_nonlinear(data_real_svm_nonlinear)
        barplot_svm_linear(data_real_svm_linear)

        reviews_data.append(
            {
                "review": "cobaa",
                "jumlah_data_clean_nb": jumlah_data_clean_nb,
                "overall_accuracy_nb": overall_accuracy_nb,
                "cr_nb": cr_nb,
                "cm_nb": cm_nb,
                "jumlah_data_clean_svm_linear": jumlah_data_clean_svm_linear,
                "overall_accuracy_svm_linear": overall_accuracy_svm_linear,
                "cr_svm_linear": cr_svm_linear,
                "cm_svm_linear": cm_svm_linear,
                "jumlah_data_clean_svm_nonlinear": jumlah_data_clean_svm_nonlinear,
                "overall_accuracy_svm_nonlinear": overall_accuracy_svm_nonlinear,
                "cr_svm_nonlinear": cr_svm_nonlinear,
                "cm_svm_nonlinear": cm_svm_nonlinear,
            }
        )

    return render_template("index.html", reviews_data=reviews_data)


if __name__ == "__main__":
    app.run(debug=True)
