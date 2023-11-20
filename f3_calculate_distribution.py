import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm, gamma
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)


def convert_label_to_num(label):
    L = ['rest', 'walk', 'run', 'dribble', 'pass', 'cross', 'shot', 'tackle']
    return L.index(label)


def size_action(action):
    return len(action)


def total_time(L):
    return len(L) * (1 / 50)


def mean_norm(L):
    return np.mean(L)


def std_norm(L):
    return np.std(L)


data1 = "match/match_1.json"
data2 = "match/match_2.json"
df1 = pd.read_json(data1)
df2 = pd.read_json(data2)
df = pd.concat([df1, df2])
df = df.drop(df[df['label'] == 'no action'].index)
df["size"] = df["norm"].apply(size_action)
df["temps_action"] = df["norm"].apply(total_time)
df["temps_tot"] = df["temps_action"].cumsum()
df["norm_mean"] = df["norm"].apply(mean_norm)
df["norm_std"] = df["norm"].apply(std_norm)

dict_limit = {'walk': 100, 'rest': 1000, 'run': 1000, 'tackle': 140, 'dribble': 130, 'pass': 150, 'cross': 120,
              'shot': 260}

dict_action = {action: [] for action in df["label"].unique()}
for action in df["label"].unique():
    df_action = df[df["label"] == action]
    df_action.reset_index(drop=True)
    for k in range(len(df_action)):
        for elmt in df_action["norm"].iloc[k]:
            dict_action[action].append(elmt)

df_action = pd.DataFrame.from_dict(dict_action, orient="index").transpose()
print(df_action)
plt.figure()
df_action.boxplot(column=['walk', 'rest', 'run', 'tackle', 'dribble', 'pass', 'cross', 'shot'])
plt.show(block=False)

dict_action_filtered = {action: [] for action in df["label"].unique()}
print(dict_action_filtered)
for action in df["label"].unique():
    df_action = df[df["label"] == action]
    df_action.reset_index(drop=True)
    for k in range(len(df_action)):
        for elmt in df_action["norm"].iloc[k]:
            if elmt < dict_limit[action]:
                dict_action_filtered[action].append(elmt)

dict_size = {"action": [], "alpha": [], "loc": [], "beta": [], "mu": [], "std": [], "min": [], "max": []}
dict_norm = {"action": [], "alpha": [], "loc": [], "beta": [], "mu": [], "std": [], "min": [], "max": []}


def distrib(action):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    data_size = df[df["label"] == action]["size"]
    ax1.hist(data_size, bins=10, density=True, alpha=0.6, color='g')
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data_size)
    mu, std = norm.fit(data_size)
    dict_size["action"].append(action)
    dict_size["alpha"].append(fit_alpha)
    dict_size["loc"].append(fit_loc)
    dict_size["beta"].append(fit_beta)
    dict_size["mu"].append(mu)
    dict_size["std"].append(std)
    dict_size["min"].append(data_size.min())
    dict_size["max"].append(data_size.max())
    xmin, xmax = ax1.set_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax1.plot(x, p, 'k', linewidth=2)
    p2 = gamma.pdf(x, fit_alpha, fit_loc, fit_beta)
    ax1.plot(x, p2, 'g', linewidth=2)
    title_size = f"Taille des {action}: mu = %.2f,  std = %.2f" % (mu, std)
    ax1.set_title(title_size)

    data_norm = dict_action_filtered[action]
    ax2.hist(data_norm, bins=60, density=True, alpha=0.6, color='red')
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data_norm)
    mu, std = norm.fit(data_norm)
    dict_norm["action"].append(action)
    dict_norm["alpha"].append(fit_alpha)
    dict_norm["loc"].append(fit_loc)
    dict_norm["beta"].append(fit_beta)
    dict_norm["mu"].append(mu)
    dict_norm["std"].append(std)
    dict_norm["min"].append(np.min(data_norm))
    dict_norm["max"].append(np.max(data_norm))
    xmin, xmax = ax2.set_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax2.plot(x, p, 'k', linewidth=2)
    p2 = gamma.pdf(x, fit_alpha, fit_loc, fit_beta)
    ax2.plot(x, p2, 'blue', linewidth=2)
    title_norm = f"Norm des {action}: mu = %.2f,  std = %.2f" % (mu, std)
    ax2.set_title(title_norm)


# distribution de la taille de chaque actions
for action in df["label"].unique():
    distrib(action)

print(dict_norm)
print(dict_size)

df_size = pd.DataFrame.from_dict(dict_size)
df_norm = pd.DataFrame.from_dict(dict_norm)

print(df_size)
print(df_norm)

df_size.to_csv(r'out/size_distribution.csv', index=False)
df_norm.to_csv(r'out/norm_distribution.csv', index=False)

plt.show(block=True)
