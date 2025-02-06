from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from io import BytesIO

def semua(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        return img_buffer

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

    return plot_cloud(wordcloud)

def positif(data):
    def plot_cloud(wordcloud):
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        img_buffer = BytesIO()
        img_buffer.seek(0)

        return img_buffer

    netral_tweets = data[data['Label NB Average'] == 'positif'].fillna('')

    netral_words = ' '.join([tweets for tweets in netral_tweets['text_tokens_stemmed']])

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        random_state=3,
        background_color='black',
        colormap='Blues_r',
        collocations=False,
        stopwords=STOPWORDS
    ).generate(netral_words)

    return plot_cloud(wordcloud)