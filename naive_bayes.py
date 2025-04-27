from sklearn.naive_bayes import MultinomialNB  # Mengimpor model Naive Bayes Multinomial dari sklearn
from sklearn.model_selection import KFold  # Mengimpor K-Fold cross-validation dari sklearn
from sklearn.metrics import accuracy_score  # Mengimpor metrik untuk menghitung akurasi
from sklearn.metrics import classification_report, confusion_matrix  # Mengimpor laporan klasifikasi dan matriks kebingungan
import pandas as pd  # Mengimpor library pandas untuk manipulasi data
import numpy as np  # Mengimpor library numpy untuk operasi numerik

def naive_bayes(A_tfid, B, C_tfid, A_fit_tfid, data_clean, data_real):  # Mendefinisikan fungsi naive_bayes
    nb = MultinomialNB()  # Membuat instance model Multinomial Naive Bayes
    nb.fit(A_tfid, B)  # Melatih model dengan data fitur (A_tfid) dan label (B)

    B_pred = nb.predict(A_tfid)  # Memprediksi label untuk data fitur A_tfid
    data_clean["Label NB"] = B_pred  # Menambahkan kolom "Label NB" ke data_clean dengan hasil prediksi
    data_clean = pd.DataFrame(data_clean)  # Mengonversi data_clean menjadi DataFrame pandas

    C_pred = nb.predict(C_tfid)  # Memprediksi label untuk data fitur C_tfid
    data_real["Label NB"] = C_pred  # Menambahkan kolom "Label NB" ke data_real dengan hasil prediksi
    data_real = pd.DataFrame(data_real)  # Mengonversi data_real menjadi DataFrame pandas
    data_clean.to_csv("sample_data/the_data_hasil_sentimen_NB.csv", index=False)  # Menyimpan data_clean ke file CSV
    data_real.to_csv("sample_data/the_data_all_hasil_sentimen_NB.csv", index=False)  # Menyimpan data_real ke file CSV
    jumlah_data_clean_nb = str(data_clean.shape[0])  # Menghitung jumlah baris data_clean dan mengonversinya ke string

    model = MultinomialNB()  # Membuat instance model Multinomial Naive Bayes baru
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Membuat objek K-Fold dengan 5 lipatan, data diacak, dan seed tetap

    fold_accuracies = []  # Membuat list kosong untuk menyimpan akurasi setiap lipatan
    all_predictions = np.zeros_like(B)  # Membuat array kosong untuk menyimpan semua prediksi
    all_true_labels = np.zeros_like(B)  # Membuat array kosong untuk menyimpan semua label sebenarnya

    for train_index, test_index in kf.split(A_tfid):  # Melakukan iterasi untuk setiap lipatan K-Fold
        A_train, A_test = A_tfid[train_index], A_tfid[test_index]  # Membagi data fitur menjadi data latih dan uji
        B_train, B_test = B.iloc[train_index], B.iloc[test_index]  # Membagi label menjadi data latih dan uji

        model.fit(A_train, B_train)  # Melatih model dengan data latih

        B_pred = model.predict(A_test)  # Memprediksi label untuk data uji

        accuracy = accuracy_score(B_test, B_pred)  # Menghitung akurasi untuk lipatan ini
        fold_accuracies.append(accuracy)  # Menambahkan akurasi ke list fold_accuracies

        all_predictions[test_index] = B_pred  # Menyimpan prediksi untuk data uji di array all_predictions
        all_true_labels[test_index] = B_test  # Menyimpan label sebenarnya di array all_true_labels

    overall_accuracy = accuracy_score(B, all_predictions)  # Menghitung akurasi keseluruhan
    cr = classification_report(all_true_labels, all_predictions)  # Membuat laporan klasifikasi
    cm = confusion_matrix(all_true_labels, all_predictions)  # Membuat matriks kebingungan
    return overall_accuracy, cr, cm, data_clean, data_real, jumlah_data_clean_nb  # Mengembalikan hasil

