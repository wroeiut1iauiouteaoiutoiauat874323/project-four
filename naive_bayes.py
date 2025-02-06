from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


def naive_bayes(A_tfid, B, C_tfid, A_fit_tfid, data_clean, data_real):
    nb = MultinomialNB()
    nb.fit(A_tfid, B)

    B_pred = nb.predict(A_tfid)
    data_clean["Label NB"] = B_pred
    data_clean = pd.DataFrame(data_clean)

    C_pred = nb.predict(C_tfid)
    data_real["Label NB"] = C_pred
    data_real = pd.DataFrame(data_real)

    data_clean.to_csv("sample_data/the_data_hasil_sentimen_NB.csv", index=False)
    data_real.to_csv("sample_data/the_data_all_hasil_sentimen_NB.csv", index=False)

    model = MultinomialNB()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    all_predictions = np.zeros_like(B)
    all_true_labels = np.zeros_like(B)

    for train_index, test_index in kf.split(A_tfid):
        A_train, A_test = A_tfid[train_index], A_tfid[test_index]
        B_train, B_test = B.iloc[train_index], B.iloc[test_index]

        model.fit(A_train, B_train)

        B_pred = model.predict(A_test)

        accuracy = accuracy_score(B_test, B_pred)
        fold_accuracies.append(accuracy)

        all_predictions[test_index] = B_pred
        all_true_labels[test_index] = B_test

    overall_accuracy = accuracy_score(B, all_predictions)
    cr = classification_report(all_true_labels, all_predictions)
    cm = confusion_matrix(all_true_labels, all_predictions)


    data_clean["Label NB Average"] = all_predictions
    data_clean.to_csv("sample_data/the_data_NB_average.csv", index=False)

    return overall_accuracy, cr, cm, data_clean
