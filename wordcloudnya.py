from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO

def wordcloud_semua_nb(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_semua_nb.png', format='png', bbox_inches='tight', pad_inches=0)

    all_words = ' '.join([tweets for tweets in data['text_tokens_stemmed'].fillna('')])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(all_words)

    plot_cloud(wordcloud)

def wordcloud_positif_nb(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_positif_nb.png', format='png', bbox_inches='tight', pad_inches=0)

    positif_tweets = data[data['Label NB'] == 'positif'].fillna('')

    positif_words = ' '.join([tweets for tweets in positif_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(positif_words)

    plot_cloud(wordcloud)

def wordcloud_negatif_nb(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_negatif_nb.png', format='png', bbox_inches='tight', pad_inches=0)

    negatif_tweets = data[data['Label NB'] == 'negatif'].fillna('')

    negatif_words = ' '.join([tweets for tweets in negatif_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(negatif_words)

    plot_cloud(wordcloud)

def wordcloud_semua_svm_nonlinear(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_semua_svm_nonlinear.png', format='png', bbox_inches='tight', pad_inches=0)

    all_words = ' '.join([tweets for tweets in data['text_tokens_stemmed'].fillna('')])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(all_words)

    plot_cloud(wordcloud)

def wordcloud_positif_svm_nonlinear(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_positif_svm_nonlinear.png', format='png', bbox_inches='tight', pad_inches=0)

    positif_tweets = data[data['Label SVM'] == 'positif'].fillna('')

    positif_words = ' '.join([tweets for tweets in positif_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(positif_words)

    plot_cloud(wordcloud)

def wordcloud_negatif_svm_nonlinear(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_negatif_svm_nonlinear.png', format='png', bbox_inches='tight', pad_inches=0)

    negatif_tweets = data[data['Label SVM'] == 'negatif'].fillna('')

    negatif_words = ' '.join([tweets for tweets in negatif_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(negatif_words)

    plot_cloud(wordcloud)

def wordcloud_semua_svm_linear(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_semua_svm_linear.png', format='png', bbox_inches='tight', pad_inches=0)

    all_words = ' '.join([tweets for tweets in data['text_tokens_stemmed'].fillna('')])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(all_words)

    plot_cloud(wordcloud)

def wordcloud_positif_svm_linear(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_positif_svm_linear.png', format='png', bbox_inches='tight', pad_inches=0)

    positif_tweets = data[data['Label SVM'] == 'positif'].fillna('')

    positif_words = ' '.join([tweets for tweets in positif_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(positif_words)

    plot_cloud(wordcloud)

def wordcloud_negatif_svm_linear(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8), tight_layout=True)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig('static/wordcloud_negatif_svm_linear.png', format='png', bbox_inches='tight', pad_inches=0)

    negatif_tweets = data[data['Label SVM'] == 'negatif'].fillna('')

    negatif_words = ' '.join([tweets for tweets in negatif_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(negatif_words)

    plot_cloud(wordcloud)

