from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def svm_classifier_nonlinear(A_tfid, B, C_tfid, A_fit_tfid, data_clean, data_real):
    # Konversi label string menjadi angka
    label_encoder = LabelEncoder()
    B_encoded = label_encoder.fit_transform(B)  # Ubah 'negatif', 'positif', dll. menjadi angka

    # Buat model SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(A_tfid, B_encoded)

    # Prediksi untuk data pelatihan
    B_pred = svm.predict(A_tfid)
    B_pred_labels = label_encoder.inverse_transform(B_pred)  # Kembalikan ke string label

    data_clean['Label SVM'] = B_pred_labels
    data_clean = pd.DataFrame(data_clean)

    # Prediksi untuk data uji
    C_pred = svm.predict(C_tfid)
    C_pred_labels = label_encoder.inverse_transform(C_pred)  # Kembalikan ke string label

    data_real['Label SVM'] = C_pred_labels
    data_real = pd.DataFrame(data_real)

    # Simpan hasil ke file CSV
    data_clean.to_csv('sample_data/the_data_hasil_sentimen_SVM_nonlinear.csv', index=False)
    data_real.to_csv('sample_data/the_data_all_hasil_sentimen_SVM_nonlinear.csv', index=False)
    jumlah_data_clean_svm = str(data_clean.shape[0])

    # K-Fold Cross Validation
    model = SVC(probability=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    all_predictions = np.zeros_like(B_encoded)
    all_true_labels = np.zeros_like(B_encoded)

    for train_index, test_index in kf.split(A_tfid):
        A_train, A_test = A_tfid[train_index], A_tfid[test_index]
        B_train, B_test = B_encoded[train_index], B_encoded[test_index]

        model.fit(A_train, B_train)
        B_pred = model.predict(A_test)

        accuracy = accuracy_score(B_test, B_pred)
        fold_accuracies.append(accuracy)

        all_predictions[test_index] = B_pred
        all_true_labels[test_index] = B_test

    overall_accuracy = accuracy_score(B_encoded, all_predictions)
    cr = classification_report(all_true_labels, all_predictions)
    cm = confusion_matrix(all_true_labels, all_predictions)

    return overall_accuracy, cr, cm, data_clean, data_real, jumlah_data_clean_svm
