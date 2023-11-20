import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import keras
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)

N_MAX_ACTIONS = 7742


def convert_num_to_label(l_action_num):
    L = ['rest', 'walk', 'run', 'dribble', 'pass', 'cross', 'shot', 'tackle']
    L_action_str = []
    for elmt in l_action_num:
        L_action_str.append(L[elmt])
    return L_action_str


def generation_seq(seq_pred, model, n_past):
    x_input = np.array([seq_pred])
    while len(seq_pred) < N_MAX_ACTIONS:
        yhat = model.predict(x_input, verbose=0)
        yhat_int = []
        for elemt in yhat[0]:
            elemt = int(elemt)
            if elemt < 0:
                yhat_int.append(0)
            elif elemt > 7:
                yhat_int.append(7)
            else:
                yhat_int.append(elemt)
        seq_pred = np.append(seq_pred, yhat_int)
        x_input = np.array([seq_pred[-n_past:]])
    return seq_pred


def generate_sequence():
    # [[[<---past--->][<---future--->]]]
    # [[[<--- Data (window size) --->]]]
    n_past = 20
    n_future = 5
    n_features = 1

    # load a seq first example, from a real one
    df = pd.read_csv('out/seq_data_0.csv')
    seq = np.array(df["action_0"])
    seq_first = seq[0:n_past]

    # load the model of generation
    model = keras.models.load_model('models/CNN_LSTM_Weights.keras')
    print(model.summary())

    seq_first_0 = np.array(seq_first)
    seq_first_0_choice = np.array(seq_first_0)
    # TROUVE RUN MOYEN D'AVOIR UN PROFIL PLUS ATTAQUANT /!\
    # bruiter la sequence d'entrée classique (match_1)
    N = len(seq_first_0)
    for k in range(N):
        i = random.randint(0, N - 1)
        elemt = random.choice(seq_first_0_choice)
        seq_first_0[i] = int(elemt)

    # generate seq from a seq first
    seq_pred_0 = generation_seq(seq_first_0, model, n_past)

    # plot the seq generated
    """plt.plot(seq, label="reel")
    plt.plot(seq_pred_0, color="r", label="predict_0")
    plt.legend()
    plt.show(block=False)"""

    # convert the seq in real labels of a match
    seq_pred_0 = convert_num_to_label(seq_pred_0)
    return seq_pred_0, seq_first_0
