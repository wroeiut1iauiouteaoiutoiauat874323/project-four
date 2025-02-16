def labelling(my_df):
    def pelabelan(skor):
        if skor < 3:
            return 'negatif'
        elif skor == 4:
            return 'positif'
        elif skor == 5:
            return 'positif'

    my_df['Label'] = my_df['score'].apply(pelabelan)
    my_df = my_df.dropna()

    my_df.to_csv('sample_data/the_data_hasillabeling.csv', index=False)

    return my_df

