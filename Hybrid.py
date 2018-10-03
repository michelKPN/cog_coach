import numpy as np
import pandas as pd
import random
import itertools
from MatrixFactorization import MF
from ContentBased import CB
# from UserDictionary import make_dict


class Hybrid:

    def __init__(self, items, ratings, features, K):
        """
                Perform content-based filtering to predict empty
                entries in a matrix and make a top 10.

                Arguments
                - items         : content of items
                - ratings       : ratings of one user
                - features      : feature influences
                - K             : number of features
                """

        self.items = items
        self.ratings = ratings
        self.features = features
        self.K = K


    def get_advice(self):

        advice = "Hi, this will be the advice!"  # 'Hi ' + name + ', ' + random.choice(top10)
        return advice
