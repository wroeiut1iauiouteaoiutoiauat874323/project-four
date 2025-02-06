import re
import nltk

def preprocessing(datanya):
    my_df = datanya[['score', 'content']]
    my_df = my_df.dropna(subset=['content'])
    my_df = casefolding(my_df)
    my_df = filtering(my_df)
    my_df = tokenizing(my_df)
    my_df = stemming(my_df)
    my_df.to_csv('sample_data/the_data_hasiltextpreprocessing.csv', index=False)

def preprocessing_all(datanya):
    my_df = datanya[['userName', 'score', 'at', 'content']].sort_values(by=['at'], ascending=False)
    my_df = my_df.dropna(subset=['content'])


def casefolding(my_df):
    def clean_text(df, text_field, new_text_field_name):
        my_df[new_text_field_name] = my_df[text_field].str.lower()
        my_df[new_text_field_name] = my_df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
        my_df[new_text_field_name] = my_df[new_text_field_name].str.replace(r"\d+", "")
        return my_df
    my_df['text_clean'] = my_df['content'].str.lower()
    data_clean = clean_text(my_df, 'content', 'text_clean')

    return data_clean

def filtering(data_clean):
    nltk.download('stopwords')
    from nltk.corpus import stopwords

    stop = stopwords.words('indonesian')
    data_clean['text_Stopword'] = data_clean['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return data_clean

def tokenizing(data_clean):
    nltk.download('punkt')
    nltk.download('punkt_tab')
    from nltk.tokenize import word_tokenize, sent_tokenize

    data_clean['text_tokens'] = data_clean['text_Stopword'].apply(lambda x: word_tokenize(x))

    return data_clean

def stemming(data_clean):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def stemmed_wrapper(term):
        return stemmer.stem(term)

    term_dict = {}
    hitung = 0

    for document in data_clean['text_tokens']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = ' '

    for term in term_dict:
        term_dict[term] = stemmed_wrapper(term)
        hitung += 1

    def get_stemmed_term(document):
        return [term_dict[term] for term in document]

    data_clean['text_tokens_stemmed'] = data_clean['text_tokens'].apply(lambda x:' '.join(get_stemmed_term(x)))

    return data_clean
