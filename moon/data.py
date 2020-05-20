from ast import literal_eval
import pandas as pd

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