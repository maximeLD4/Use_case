import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from f2_generate_sequence import generate_sequence
import sys

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.random.seed(42)


def total_time(L):
    return len(L) * (1 / 50)


def convert_minutes(time_sec):
    return time_sec/60


def generate_match(liste_actions, n_secondes):
    dict_gen = {'label': [], 'norm': []}
    temps_tot = 0
    k = 0
    for elmt in liste_actions:
        if temps_tot > n_secondes:
            return dict_gen
        size = -1
        label = elmt
        L_norme = []
        while size < np.array(df_size[df_size["action"] == label]["min"])[0] or \
                size > np.array(df_size[df_size["action"] == label]["max"])[0]:
            size = \
                np.random.normal(df_size[df_size["action"] == label]["mu"],
                                 df_size[df_size["action"] == label]["std"], 1)[0]
        temps_tot += size * (1 / 50)
        for k in range(int(size)):
            norme = -1
            while np.array(df_norm[df_norm["action"] == label]["min"])[0] > norme or \
                    np.array(df_norm[df_norm["action"] == label]["max"])[0] < norme:
                # norme = np.round(np.random.gamma(df_norm[df_norm["action"] == action]["alpha"], df_norm[df_norm["action"] == action]["beta"], 1)[0], 3)
                norme = \
                    np.random.gamma(df_norm[df_norm["action"] == label]["alpha"],
                                    df_norm[df_norm["action"] == label]["beta"],
                                    1)[0]
            L_norme.append(norme)
        dict_gen['norm'].append(L_norme)
        dict_gen['label'].append(liste_actions[k])
        k += 1
    return dict_gen


if len(sys.argv) != 2:
    print("must use the following command : python3.10 f4_generate_match.py minutes_de_jeu")
    exit(1)
number_secondes = int(sys.argv[1]) * 60

df_size = pd.read_csv('out/size_distribution.csv')
df_norm = pd.read_csv('out/norm_distribution.csv')

liste_actions, seq_first = generate_sequence()

dict_generation = generate_match(liste_actions, number_secondes)

dataframe_generation = pd.DataFrame.from_dict(dict_generation, orient="index").transpose()
dataframe_generation["total_time"] = dataframe_generation['norm'].apply(total_time)
dataframe_generation["total_time"] = dataframe_generation["total_time"].cumsum()
dataframe_generation["total_time_minutes"] = dataframe_generation["total_time"].apply(convert_minutes)
print(dataframe_generation)

dataframe_generation = dataframe_generation[["label", "norm"]]
result = dataframe_generation.to_json(orient="records")
with open(f'match/match_created_{str(seq_first)}.json', 'w') as f:
    f.write(result)

plt.show(block=True)
