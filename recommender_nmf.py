from sklearn.decomposition import NMF
import pickle
import numpy as np
import pandas as pd


Q_mat = pd.read_csv("Q_mat.csv", index_col=0)
with open('myfilms.pickle', "rb") as f:
    films_col = pickle.load(f)


def recommend_nmf(query,
                  model="my_nmf_model.sav",
                  films = films_col,
                  Q_matrix=Q_mat,
                  k=10):
    """Filters and recommends the top k movies
    for any given input query based
    on a trained NMF model.

    Parameters
    ----------
    query : dict
        A dictionary of movies already seen.
        Takes the form {"movie_A": 3, "movie_B": 3} etc
    model : pickle
        pickle model read from disk
    k : int, optional
        no. of top movies to recommend, by default 10
    """

    films_dict = dict.fromkeys(films, np.nan)
    for key in query:
        if key!="":
            films_dict[key]=query[key]
    ###Dictionary
    user_ratings = films_dict
    #dict to df
    user_ratings = pd.DataFrame(user_ratings, index=[0])
    #copy from df
    user_original = user_ratings.copy()
    #model
    model = pickle.load(open(f'./{model}', 'rb'))
    #
    user_ratings.fillna(0, inplace=True)
    #
    P_user = model.transform(user_ratings)
    #
    R_user = np.dot(P_user, Q_matrix)
    #
    user_df = pd.DataFrame(R_user, columns=films)
    #
    boolean_mark = user_original.isna()
    #
    unrated_movies_df = user_df[boolean_mark]
    #
    sorted_new_user_df = unrated_movies_df.T.sort_values(by=0, ascending=False)

    top_k = sorted_new_user_df[:k]
    return list(top_k.index)


if __name__== '__main__':
    query = {"'Hellboy': The Seeds of Creation (2004)": 4,
        'Nixon (1995)': 2.5,
        'A Street Cat Named Bob (2016)': 2.5,
        'Afro Samurai (2007)': 5}
    top_k = recommend_nmf(query = query)
    print(top_k)
