import random
import tkinter as tk

import numpy as np
import pandas as pd

DATAFOLDER = "C:/Users/kleis500/Eclipse workspace/RecSys/data/cog_coach/"

# Read in the data
df = pd.read_excel(DATAFOLDER + "acties_features_rated.xlsx")

X = []
for index, row2 in df.iterrows():
    X.append([row2["binnen"], row2["buiten"], row2["alleen"], row2["samen"], row2["kort"], row2["lang"]])

df["Item Profile"] = X

# Anja
# Add column with indicator for rating (1 = rating, 0 = no rating)
rij = []
for row in df["Anja"]:
    if row > 0:
        rij.append(1)
    else:
        rij.append(0)

# df"r_ij"] = df["Anja"].apply(lambda x: np.isnan(x) and 0 or 1)
df["Rating"] = rij

# Make new dataframe with rated items
anja_matrix = df.loc[df['Rating'] == 1]

summed_profile = [0, 0, 0, 0, 0, 0]
for i in anja_matrix["Item Profile"]:
    count = 0
    for j in i:
        summed_profile[count] = summed_profile[count] + j
        count += 1

# calculate star rating (weight for item profiles)
# count number of feature occurence for 1 user
am_binnen = anja_matrix['binnen'].loc[anja_matrix['binnen'] > 0].count()
am_buiten = anja_matrix['buiten'].loc[anja_matrix['buiten'] > 0].count()
am_alleen = anja_matrix['alleen'].loc[anja_matrix['alleen'] > 0].count()
am_samen = anja_matrix['samen'].loc[anja_matrix['samen'] > 0].count()
am_kort = anja_matrix['kort'].loc[anja_matrix['kort'] > 0].count()
am_lang = anja_matrix['lang'].loc[anja_matrix['lang'] > 0].count()
count_anja = [am_binnen, am_buiten, am_alleen, am_samen, am_kort, am_lang]

# Extract raw ratings for user per feature
am_binnen_rating = list(anja_matrix['Anja'].loc[anja_matrix['binnen'] > 0])
am_buiten_rating = list(anja_matrix['Anja'].loc[anja_matrix['buiten'] > 0])
am_alleen_rating = list(anja_matrix['Anja'].loc[anja_matrix['alleen'] > 0])
am_samen_rating = list(anja_matrix['Anja'].loc[anja_matrix['samen'] > 0])
am_kort_rating = list(anja_matrix['Anja'].loc[anja_matrix['kort'] > 0])
am_lang_rating = list(anja_matrix['Anja'].loc[anja_matrix['lang'] > 0])

raw_rating_list = [am_binnen_rating, am_buiten_rating, am_alleen_rating, am_samen_rating, am_kort_rating,
                   am_lang_rating]

# Normalizae raw ratings for user
normalized_ratings = []
for i in raw_rating_list:
    temp_list = []
    for j in i:
        j -= 3
        temp_list.append(j)
    normalized_ratings.append(temp_list)

# Sum normalized ratings per feature
sum_ratings = []
for i in normalized_ratings:
    sum_ratings.append(sum(i))

weights_anja = np.array(sum_ratings) / np.array(count_anja)
summed_weighted_profiles = weights_anja * summed_profile
# Personalized user profiles
weighted_anja_user_profile = summed_weighted_profiles / np.array([len(df), len(df), len(df), len(df), len(df), len(df)])

similarities = []
# Compute the cosine similarity between item profile and user profile
for item in df['Item Profile']:
    cosine_sim = (weighted_anja_user_profile * item) / (
            np.linalg.norm(weighted_anja_user_profile) * np.linalg.norm(item))

    #     cosine_sim = linear_kernel(weighted_anja_user_profile, item)
    similarities.append(cosine_sim)

summed_cos = []
for item in similarities:
    summed_cos.append(sum(item))

df['summed_cos'] = summed_cos

predictions = df.sort_values(by='summed_cos', ascending=False)

top5 = list(predictions["Actie"][:5])
advice = 'Hi Anja, ' + random.choice(top5)

window = tk.Tk()

window.title('Jouw Cognitive Coach zegt: ')

lbl = tk.Label(window, text='Hi Anja, ' + random.choice(top5), font=('Arial', 18), bg="green", fg='white')
lbl.grid(column=5, row=5)
window.geometry('1000x100')
window.mainloop()
window.quit()
