import matplotlib.pyplot as plt
import pandas as pd

def barplot_nb(dr):
    # Convert 'at' column to datetime and sort by date
    dr['date'] = pd.to_datetime(dr['at'])
    dr = dr.sort_values(by='date')

    # Format the date to 'YYYY-MM-DD' and convert back to datetime
    dr['date'] = dr['date'].dt.strftime('%Y-%m-%d')
    dr['date'] = pd.to_datetime(dr['date'])

    # Calculate value counts and percentages
    value_counts = dr['Label NB'].value_counts()
    percentages = value_counts / value_counts.sum() * 100

    # Plot bar chart with percentages
    ax = value_counts.plot(kind='bar')
    for i, (count, percentage) in enumerate(zip(value_counts, percentages)):
        ax.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.title('Sentiment Analysis menggunakan Algoritma Naive Bayes')
    plt.xlabel('Sentiment')
    plt.ylabel('Total')

    # Save the plot as an image file
    plt.savefig('static/barplot_nb.png', format='png')

def barplot_svm(dr):
    # Convert 'at' column to datetime and sort by date
    dr['date'] = pd.to_datetime(dr['at'])
    dr = dr.sort_values(by='date')

    # Format the date to 'YYYY-MM-DD' and convert back to datetime
    dr['date'] = dr['date'].dt.strftime('%Y-%m-%d')
    dr['date'] = pd.to_datetime(dr['date'])

    # Calculate value counts and percentages
    value_counts = dr['Label SVM'].value_counts()
    percentages = value_counts / value_counts.sum() * 100

    # Plot bar chart with percentages
    ax = value_counts.plot(kind='bar')
    for i, (count, percentage) in enumerate(zip(value_counts, percentages)):
        ax.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom')

    plt.title('Sentiment Analysis menggunakan Algoritma SVM')
    plt.xlabel('Sentiment')
    plt.ylabel('Total')

    # Save the plot as an image file
    plt.savefig('static/barplot_svm.png', format='png')