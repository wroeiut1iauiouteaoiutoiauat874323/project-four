from sklearn.naive_bayes import MultinomialNB
import pandas as pd

def naive_bayes(A_tfid, B, C_tfid, A_fit_tfid, data_clean, data_real):
    nb = MultinomialNB()
    nb.fit(A_tfid, B)

    B_pred = nb.predict(A_tfid)
    data_clean['Label NB'] = B_pred
    data_clean = pd.DataFrame(data_clean)
    data_clean.to_csv('sample_data/the_data_hasil_sentimen_NB.csv', index=False)
