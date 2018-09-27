import numpy as np
import pandas as pd
import random


class CB:

    def __init__(self, items, ratings, features):
        """
                Perform content-based filtering to predict empty
                entries in a matrix and make a top 10.

                Arguments
                - items         : content of items
                - ratings       : ratings of one user
                - features      : feature influences
                """

        self.items = items
        self.ratings = ratings
        self.features = features

    def make_profiles(self, person, name):
        # Get the ratings of the indicated person
        person_ratings = self.ratings.iloc[:, [int(person)]].copy()

        # Make one DataFrame from the ratings, features and items
        data = [self.items, person_ratings, self.features]
        df = pd.concat(data, axis=1)

        # Compute item profiles
        item_profiles = self.item_profile()
        df["Item Profile"] = item_profiles

        # Add column with indicator for rating (1 = rating, 0 = no rating)
        rij = 0
        bool_rating = [rij+1 if x > 0 else rij+0 for x in person_ratings[name]]
        df["Rating"] = bool_rating
        user_matrix = df.loc[df['Rating'] == 1]

        # Compute user profile
        user_profile = self.user_profile(user_matrix, name)

        return item_profiles, user_profile

    def item_profile(self):
        # Make vector of the features per action
        item_profiles = []
        for index, row in self.features.iterrows():
            item_profiles.append([row[0], row[1], row[2], row[3], row[4], row[5]])

        return item_profiles

    def user_profile(self, user_matrix, name):
        # Count number of feature occurrences
        feature_occ = []
        for f in self.features.columns.values.tolist():
            feature_occ.append(user_matrix[f].loc[user_matrix[f] > 0].count())

        # Extract raw ratings for user per feature
        feature_raw = []
        for f in self.features.columns.values.tolist():
            feature_raw.append(list(user_matrix[name].loc[user_matrix[f] > 0]))

        # Normalize all raw ratings
        normalized_ratings = [[j-3 for j in i] for i in feature_raw]

        # Sum normalized ratings per feature
        sum_norm_ratings = [sum(k) for k in normalized_ratings]

        # Compute profile weights
        profile_weights = np.array(sum_norm_ratings) / np.array(feature_occ)

        # Multiply weights with feature occurrences ??? or summed feature inputs ???
        summed_weighted_profiles = profile_weights * feature_occ

        # Create personalized user profile
        weighted_user_profile = summed_weighted_profiles / np.array([len(self.items)] * 6)

        return weighted_user_profile

    def cosine_similarity(self, item_profiles, user_profile):
        # Compute the cosine similarity between item profile and user profile for each feature
        similarities = [((user_profile * item) / (np.linalg.norm(user_profile) * np.linalg.norm(item)))
                        for item in item_profiles]

        # Sum the feature similarity distances per item
        predictions = [sum(sim) for sim in similarities]

        return predictions

    def get_advice(self):
        # Select a person and copy the ratings
        person = input("Choose a person: Anja=0, Bert=1, Carlos=2 or Dave=3")
        print("Person number is: " + person)
        # Save person's name
        if int(person) == 0:
            name = "Anja"
        elif int(person) == 1:
            name = "Bert"
        elif int(person) == 2:
            name = "Carlos"
        elif int(person) == 3:
            name = "Dave"
        else:
            name = "user"

        # Get the item- and user profiles
        profiles = self.make_profiles(person, name)
        item_profiles = profiles[0]
        user_profile = profiles[1]

        # Compute the cosine similarity between the user profile and item profiles
        predictions = self.cosine_similarity(item_profiles, user_profile)

        # Link the predictions to the items by placing them in a DataFrame
        item_predictions = pd.DataFrame()
        item_predictions["Item"] = self.items
        item_predictions["Prediction"] = predictions

        # Sort the predictions by summed cosine similarity
        sorted_pred = item_predictions.sort_values(by='Prediction', ascending=False)

        # Make top 5 and choose one random advice
        top5 = list(sorted_pred["Item"][:5])
        advice = 'Hi ' + name + ', ' + random.choice(top5)

        return advice





