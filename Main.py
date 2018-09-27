import numpy as np
import pandas as pd
from MatrixFactorization import MF
from ContentBased import CB


def run_cb(items, ratings, features):
    print("Running cb")
    cb = CB(items, ratings, features)
    advice = cb.get_advice()
    return advice


def run_mf(ratings, items):
    print("Running mf")
    R = np.array(ratings)
    mf = MF(R, K=12, alpha=0.1, beta=0.01, iterations=30)
    advice = mf.get_advice(items)
    return advice


def main():
    print("----- Running the Cognitive Coach -----")
    data_folder = "C://Users/galet500//Documents//Data & Analytics//Data Innovation Lab//Behaviour and Cognition//" \
                  "Recommendation System//data//"
    df = pd.read_excel(data_folder + "acties_features_rated.xlsx")
    # Split DataFrame into items, ratings and features
    items = df.iloc[:, 0].copy()
    ratings = df.iloc[:, [1, 2, 3, 4]].copy()
    ratings.fillna(0, inplace=True)
    features = df.iloc[:, [5, 6, 7, 8, 9, 10]].copy()

    # Choose recommender type and run
    recommender_type = input("Which recommender do you want to use? Content-Based = 0, Matrix Factorization = 1 ")

    if int(recommender_type) == 0:
        print("type = 0")
        final_advice = run_cb(items, ratings, features)
    elif int(recommender_type) == 1:
        print("type = 1")
        final_advice = run_mf(ratings, items)
    else:
        return

    print("Advies: " + final_advice)


if __name__ == "__main__":
    main()