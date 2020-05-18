import pandas as pd
from pathlib import Path
import numpy as np

# Holds are in the form 'A18' to 'K1'; we want them as (row, col) coordinates
def _hold_to_coord(hold):
    row = 18 - int(hold[1:len(hold)])
    col = ord(hold[0].upper()) - ord('A')
    return row, col

def problems_to_array(df):
    array = np.repeat(
        [np.stack([
            np.zeros((18, 11)), np.zeros((18, 11)), np.zeros((18, 11)), np.ones((18, 11))
        ], axis=-1)],
        repeats=len(df.index), axis=0
    ).astype(np.uint8)
    cols = ['Holds.End', 'Holds.Start', 'Holds.Others']
    count = 0
    for _, problem in df.iterrows():
        for holdtype_depth, holdtype_label in enumerate(cols):
            for hold in problem[holdtype_label]:
                r, c = _hold_to_coord(hold)
                array[count, r, c, holdtype_depth] = 1
                array[count, r, c, -1] = 0
        count = count + 1
    return array