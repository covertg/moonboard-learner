from ast import literal_eval
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from moon.problem import Problem

def read_problems(file, convert_df=True):
    probs = pd.read_csv(file)
    # Remove . because itertuples() won't accept them
    probs.rename(columns=lambda s: s.replace('.', '_'), inplace=True)
    # Safely eval() the string-list literals for the holds
    probs['Holds_Start'] = probs['Holds_Start'].apply(literal_eval)
    probs['Holds_Intermed'] = probs['Holds_Intermed'].apply(literal_eval)
    probs['Holds_End'] = probs['Holds_End'].apply(literal_eval)
    # Return list of Problem objects
    if convert_df:
        probs = [Problem(t) for t in probs.itertuples()]
    return probs


# split is [train, validate, test] or [train, test]
def split_data(probs, split=[.6, .2, .2], seed=0):
    assert (len(split) == 2 or len(split) == 3)
    # Stratify-split the data first into train and other
    ranks = [p.grade.rank for p in probs]
    probs_train, probs_other = train_test_split(probs, train_size=split[0], stratify=ranks, shuffle=True, random_state=seed)
    if len(split) == 2:
        return probs_train, probs_other
    elif len(split) == 3:
        # Then into validation and test
        ranks_other = [p.grade.rank for p in probs_other]
        probs_val, probs_test = train_test_split(probs_other, train_size=split[1]/(split[1]+split[2]), stratify=ranks_other, shuffle=True, random_state=seed)
        return probs_train, probs_val, probs_test