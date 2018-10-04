import numpy as np
import pandas as pd
from MatrixFactorization import MF
from ContentBased import CB


def run_cb(items, ratings, features, amount_features):
    print("Running cb")
    cb = CB(items, ratings, features, K=amount_features)
    advice = cb.get_advice()
    return advice


def run_mf(ratings, items, amount_features):
    print("Running mf")
    R = np.array(ratings)
    mf = MF(R, K=amount_features, alpha=0.1, beta=0.01, iterations=30)
    advice = mf.get_advice(items)
    return advice


def run_hybrid(ratings, items, features, amount_features):
    print("Running hybrid")
    hb = Hybric(items, ratings, features, K=amount_features)
    advice = hb.get_advice()
    return advice


def main():
    print("----- Running the Cognitive Coach -----")
    data_folder = "C://Users/galet500//Documents//Data & Analytics//Data Innovation Lab//Behaviour and Cognition//" \
                  "Recommendation System//data//"
    #data_folder = "C:/Users/kleis500/Eclipse workspace/RecSys/data/cog_coach/"
    #data_folder = "C:/Users/hoef503/Desktop/DataLab/Behaviour & Cognition/20-8/data/"
    df = pd.read_excel(data_folder + "acties_features_rated.xlsx")
    # Split DataFrame into items, ratings and features
    items = df.iloc[:, 0].copy()
    ratings = df.iloc[:, [1, 2, 3, 4]].copy()
    ratings.fillna(ratings.mean(), inplace=True)
    features = df.iloc[:, [5, 6, 7, 8, 9, 10]].copy()

    # Choose recommender type and run
    recommender_type = input("Which recommender do you want to use? Content-Based = 0, Matrix Factorization = 1 ")
    amount_features = input("How many features do you have? ")

    if int(recommender_type) == 0:
        print("type = 0")
        final_advice = run_cb(items, ratings, features, int(amount_features))
    elif int(recommender_type) == 1:
        print("type = 1")
        final_advice = run_mf(ratings, items, int(amount_features))
    elif int(recommender_type) == 2:
        print("type = 2")
        final_advice = run_hybrid(ratings, items, features, int(amount_features))
    else:
        return

    print("Advies: " + final_advice)


if __name__ == "__main__":
    main()