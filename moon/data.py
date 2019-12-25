from collections import defaultdict
import pandas as pd
from pathlib import Path
import numpy as np
import requests
import simplejson as json

MB_REQ = 'https://moonboard.com/Problems/GetProblems'
MB_HEADER = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:71.0) Gecko/20100101 Firefox/71.0',
    'Accept': '*/*',
    'Accept-Language': 'es,en-US;q=0.7,en;q=0.3',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Origin': 'https://www.moonboard.com',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Referer': 'https://www.moonboard.com/Problems/Index',
    'Cookie': ''
}

def download_problems(cookies, pages=1843, page_len=15, wanted_keys=['Name', 'Grade', 'UserGrade', 'UserRating', 'Repeats', 'Holds.Start', 'Holds.End', 'Holds.Others']):
    MB_HEADER['Cookie'] = cookies
    def _post(page, n):
        print('POSTing for p{} (n={})'.format(page, n))
        data = { 'sort': '', 'page': str(page), 'pageSize': str(n), 'group': '', 'filter': "setupId~eq~'1'" }  # PSA. setupId determines whether we filter for 2016 problems or other setups
        r = requests.post(MB_REQ, headers=MB_HEADER, data=data)
        if not r.text or r.status_code != 200:
            print('Failed reponse on p{}. Status: {}'.format(page, r.status_code))
            return None
        return r.json()['Data']
    def _clean_problem(prob):
        # Clean up the holds list: so each move is just a 3-tuple (str coordinate, bool is_start, bool is_end)
        prob['Holds.Start'], prob['Holds.End'], prob['Holds.Others'] = [], [], []
        for m in prob['Moves']:
            if m['IsStart']: prob['Holds.Start'].append(m['Description'])
            elif m['IsEnd']: prob['Holds.End'].append(m['Description'])
            else: prob['Holds.Others'].append(m['Description'])
        # Filter by keys
        prob = {k: v for k, v in prob.items() if k in wanted_keys}
        return prob

    all_probs = []
    for i in range(pages, 0, -1):  # We go backwards to avoid duplicates (in case a user adds a problem while we're downloading)
        response = _post(i, page_len)
        if not response:  # Bad response => we're done.
            break
        for problem in response:
            all_probs.append(_clean_problem(problem))
    return all_probs


def write_problems(problems, filename='probs.json'):
    p = Path(filename)
    if p.exists():
        print('File {} already exists. Exiting'.format(filename))
        return
    with p.open('w') as f:
        json.dump(problems, f, separators=(',', ':'), iterable_as_array=True)

def read_problems(filename='probs.json', remove_dups=False):
    probs = pd.read_json(filename)
    # Ensure capitalization of holds
    probs['Holds.Start'] = probs['Holds.Start'].map(lambda x: [s.upper() for s in x])
    probs['Holds.End'] = probs['Holds.End'].map(lambda x: [s.upper() for s in x])
    probs['Holds.Others'] = probs['Holds.Others'].map(lambda x: [s.upper() for s in x])
    if remove_dups:
        dups = probs[probs.applymap(lambda x: str(sorted(x)) if isinstance(x, list) else x).duplicated(['Name', 'Holds.Start', 'Holds.End', 'Holds.Others'], keep=False)]
        print('Pruning {} duplicate probs.'.format(len(dups)))
        if get_ipython():
            with pd.option_context('display.max_rows', None, 'display.max_columns', None): display(dups)
        # Prefer problems by: higher repeats, higher UserRating, newest
        indices = dups.index
        keeps = []
        i = 0
        while i < len(indices):
            idxs = [indices[i], indices[i + 1]]
            i += 2
            while i < len(indices) and dups.loc[indices[i]]['Name'] == dups.loc[indices[i-1]]['Name']:
                idxs.append(i)
                i += 1
            repeats = dups.reindex(idxs)['Repeats']
            ratings = dups.reindex(idxs)['UserRating']
            remax, remin, ramax, ramin = repeats.idxmax(), repeats.idxmin(), ratings.idxmax(), ratings.idxmin()
            if remax != remin:
                keep = remax
            elif ramax != ramin:
                keep = ramax
            else:
                keep = idxs[-1]
            keeps.append(keep)
        drops = set(indices) - set(keeps)
        probs.drop(drops, inplace=True)
        probs.reset_index(inplace=True, drop=True)
    return probs

# Holds are in the form 'A18' to 'K1'; we want them as (row, col) coordinates
def hold_to_coord(hold):
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
                r, c = hold_to_coord(hold)
                array[count, r, c, holdtype_depth] = 1
                array[count, r, c, -1] = 0
        count = count + 1
    return array