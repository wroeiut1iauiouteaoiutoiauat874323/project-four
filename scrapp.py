import pandas as pd
import numpy as np
from google_play_scraper import Sort, reviews, reviews_all
import os


def scrapp_3000_data(id_app):

    file_path = "sample_data/the_data_selected.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=",")
    else:
        result, continuation_token = reviews(
            id_app,
            lang="id",  # defaults to 'en'
            country="id",  # defaults to 'us'
            sort=Sort.NEWEST,  # defaults to Sort.MOST_RELEVANT
            count=3000,  # defaults to 100
            filter_score_with=None,  # defaults to None(means all score) Use 1 for only 1 star, 2 for 2 star, 3 for 3 star, 4 for 4 star and 5 for 5 star.
        )

        df = pd.DataFrame(np.array(result), columns=["review"])
        df = df.join(pd.DataFrame(df.pop("review").tolist()))
        df.to_csv(file_path, index=False)

    return df

def scrapp_all_data(id_app):
    file_path = "sample_data/the_data_all_selected.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=",")
    else:
        review = reviews_all(
            id_app,
            sleep_milliseconds=0, # defaults to 0
            lang='id', # defaults to 'en'
            country='id', # defaults to 'us'
            sort=Sort.NEWEST # defaults to Sort.MOST_RELEVANT
        )

        df_jobstreet = pd.DataFrame(np.array(review), columns=['content'])
        df_jobstreet = df_jobstreet.join(pd.DataFrame(df_jobstreet.pop('content').tolist()))
        df.to_csv(file_path, index=False)

    return df