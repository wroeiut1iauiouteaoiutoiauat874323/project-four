from sklearn.svm import SVC  # Import library untuk Support Vector Classifier
from sklearn.model_selection import KFold  # Import library untuk K-Fold Cross Validation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Import library untuk evaluasi model
from sklearn.preprocessing import LabelEncoder  # Import library untuk encoding label
import pandas as pd  # Import library untuk manipulasi data
import numpy as np  # Import library untuk operasi numerik
def svm_classifier_linear(A_tfid, B, C_tfid, A_fit_tfid, data_clean, data_real):  # Definisi fungsi untuk klasifikasi SVM dengan kernel linear
    # Konversi label string menjadi angka
    label_encoder = LabelEncoder()  # Inisialisasi encoder label
    B_encoded = label_encoder.fit_transform(B)  # Ubah label string (misalnya 'negatif', 'positif') menjadi angka
    # Buat model SVM dengan kernel linear
    svm = SVC(kernel='linear', probability=True)  # Inisialisasi model SVM dengan kernel linear
    svm.fit(A_tfid, B_encoded)  # Latih model SVM menggunakan data fitur dan label
    # Prediksi untuk data pelatihan
    B_pred = svm.predict(A_tfid)  # Prediksi label untuk data pelatihan
    B_pred_labels = label_encoder.inverse_transform(B_pred)  # Konversi kembali label angka ke string
    data_clean['Label SVM'] = B_pred_labels  # Tambahkan kolom label hasil prediksi ke data_clean
    data_clean = pd.DataFrame(data_clean)  # Konversi ke DataFrame pandas
    # Prediksi untuk data uji
    C_pred = svm.predict(C_tfid)  # Prediksi label untuk data uji
    C_pred_labels = label_encoder.inverse_transform(C_pred)  # Konversi kembali label angka ke string
    data_real['Label SVM'] = C_pred_labels  # Tambahkan kolom label hasil prediksi ke data_real
    data_real = pd.DataFrame(data_real)  # Konversi ke DataFrame pandas
    # Simpan hasil ke file CSV
    data_clean.to_csv('sample_data/the_data_hasil_sentimen_SVM_linear.csv', index=False)  # Simpan data_clean ke file CSV
    data_real.to_csv('sample_data/the_data_all_hasil_sentimen_SVM_linear.csv', index=False)  # Simpan data_real ke file CSV
    jumlah_data_clean_svm = str(data_clean.shape[0])  # Hitung jumlah baris data_clean dan ubah ke string
    # K-Fold Cross Validation
    model = SVC(kernel='linear', probability=True)  # Inisialisasi model SVM untuk K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Inisialisasi K-Fold dengan 5 lipatan
    fold_accuracies = []  # List untuk menyimpan akurasi setiap lipatan
    all_predictions = np.zeros_like(B_encoded)  # Array untuk menyimpan semua prediksi
    all_true_labels = np.zeros_like(B_encoded)  # Array untuk menyimpan semua label sebenarnya
    for train_index, test_index in kf.split(A_tfid):  # Iterasi melalui setiap lipatan K-Fold
        A_train, A_test = A_tfid[train_index], A_tfid[test_index]  # Bagi data fitur menjadi data latih dan uji
        B_train, B_test = B_encoded[train_index], B_encoded[test_index]  # Bagi label menjadi data latih dan uji
        model.fit(A_train, B_train)  # Latih model pada data latih
        B_pred = model.predict(A_test)  # Prediksi label pada data uji
        accuracy = accuracy_score(B_test, B_pred)  # Hitung akurasi untuk lipatan ini
        fold_accuracies.append(accuracy)  # Tambahkan akurasi ke list
        all_predictions[test_index] = B_pred  # Simpan prediksi untuk data uji
        all_true_labels[test_index] = B_test  # Simpan label sebenarnya untuk data uji
    overall_accuracy = accuracy_score(B_encoded, all_predictions)  # Hitung akurasi keseluruhan
    cr = classification_report(all_true_labels, all_predictions)  # Buat laporan klasifikasi
    cm = confusion_matrix(all_true_labels, all_predictions)  # Buat matriks kebingungan
    return overall_accuracy, cr, cm, data_clean, data_real, jumlah_data_clean_svm  # Kembalikan hasil evaluasi dan data

