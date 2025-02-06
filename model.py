from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Data latih
training_data = [
    ("Aplikasi ini sangat bagus", "positif"),
    ("Aplikasi ini buruk", "negatif"),
    ("Saya suka aplikasi ini", "positif"),
    ("Fitur aplikasi sangat buruk", "negatif")
]

# Pisahkan teks dan label
X_train, y_train = zip(*training_data)

# Membuat pipeline Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Simpan model ke file
joblib.dump(model, 'models/naive_bayes.pkl')
