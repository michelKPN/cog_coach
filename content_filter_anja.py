import random
import tkinter as tk
import itertools
import numpy as np
import pandas as pd
## dit is een check
# hallooo annemarie
# hallooo michel

#DATAFOLDER = "C:/Users/kleis500/Eclipse workspace/RecSys/data/cog_coach/"
DATAFOLDER = "C:\\Users\\hoef503\\Desktop\\DataLab\\Behaviour & Cognition\\data\\"

#read in data
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

df["Rating"] = rij

# Make new dataframe with rated items
anja_matrix = df.loc[df['Rating'] == 1]

summed_profile = [0, 0, 0, 0, 0, 0]
for i in anja_matrix["Item Profile"]:
    count = 0
    for j in i:
        summed_profile[count] = summed_profile[count] + j
        count += 1

# Extract raw ratings for user per feature
am_binnen_rating = list(anja_matrix['Anja'].loc[anja_matrix['binnen'] > 0])
am_buiten_rating = list(anja_matrix['Anja'].loc[anja_matrix['buiten'] > 0])
am_alleen_rating = list(anja_matrix['Anja'].loc[anja_matrix['alleen'] > 0])
am_samen_rating = list(anja_matrix['Anja'].loc[anja_matrix['samen'] > 0])
am_kort_rating = list(anja_matrix['Anja'].loc[anja_matrix['kort'] > 0])
am_lang_rating = list(anja_matrix['Anja'].loc[anja_matrix['lang'] > 0])

raw_rating_list = [am_binnen_rating, am_buiten_rating, am_alleen_rating, am_samen_rating, am_kort_rating,
                   am_lang_rating]

flat_ratings = list(itertools.chain.from_iterable(raw_rating_list))
mean_ratings = sum(flat_ratings)/len(flat_ratings)

# Normalizae raw ratings for user
normalized_ratings = []
for i in raw_rating_list:
    temp_list = []
    for j in i:
        j -= mean_ratings
        temp_list.append(j)
    normalized_ratings.append(temp_list)

# Sum normalized ratings per feature
sum_ratings = []
for i in normalized_ratings:
    sum_ratings.append(sum(i))

# # Personalized user profiles
weighted_anja_user_profile = (np.array(sum_ratings) * np.array(summed_profile))/np.array([len(df), len(df), len(df), len(df), len(df), len(df)])


similarities = []
# Compute the cosine similarity between item profile and user profile
for item in df['Item Profile']:
    cosine_sim = (weighted_anja_user_profile * item) / (
            np.linalg.norm(weighted_anja_user_profile) * np.linalg.norm(item))

    #cosine_sim = linear_kernel(weighted_anja_user_profile, item)
    similarities.append(cosine_sim)

summed_cos = []
for item in similarities:
    summed_cos.append(sum(item))

df['summed_cos'] = summed_cos

df.sort_values(by='summed_cos', ascending=False, inplace = True)

top5 = list(df["Actie"][:5])
advice = 'Hi Anja, ' + random.choice(top5)

window = tk.Tk()

window.title('Jouw Cognitive Coach zegt: ')

lbl = tk.Label(window, text='Hi Anja, ' + random.choice(top5), font=('Arial', 18), bg="green", fg='white')
lbl.grid(column=5, row=5)
window.geometry('1000x100')
window.mainloop()
window.quit()
