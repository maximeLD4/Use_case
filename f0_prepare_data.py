import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def convert_label_to_num(label):
    L = ['rest', 'walk', 'run', 'dribble', 'pass', 'cross', 'shot', 'tackle']
    return L.index(label)


data1 = "match/match_1.json"
data2 = "match/match_2.json"
df1 = pd.read_json(data1)
df2 = pd.read_json(data2)

# clean unclear labels
df1.drop(df1[df1['label'] == 'no action'].index, inplace=True)
df1.reset_index(drop=True, inplace=True)
df2.drop(df2[df2['label'] == 'no action'].index, inplace=True)
df2.reset_index(drop=True, inplace=True)

# convert labels into numbers
df1["label_num"] = df1['label'].apply(convert_label_to_num)
df2["label_num"] = df2['label'].apply(convert_label_to_num)

# extract/create new dataframe, using a windows throught the data as functions
window_size = 25
seq1 = np.array(df1['label_num'])
seq2 = np.array(df2['label_num'])

dict1 = {f'action_{i}': [seq1[i + k] for k in range(len(seq1) - window_size)] for i in range(window_size)}
dict2 = {f'action_{i}': [seq2[i + k] for k in range(len(seq2) - window_size)] for i in range(window_size)}

new_df1 = pd.DataFrame.from_dict(dict1)
new_df2 = pd.DataFrame.from_dict(dict1)

# regroup data in a df
df = pd.concat([new_df1, new_df2])
# reset indexation
df = df.reset_index(drop=True)
df = df.drop_duplicates(keep='first')
df = df.reset_index(drop=True)
print("df\n", df)
df.to_csv(r'out/seq_data_0.csv', index=False)
