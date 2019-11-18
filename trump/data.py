import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

STOP_CHAR = "\u2665"
PAD_CHAR = "\u2666"
MAX_LEN = 280


def load_data():
    res = pd.read_csv("./tweets")
    return res


def process_data(data):
    res = data.copy()
    contains_http = res["text"].str.contains("http")
    drop = contains_http | res["is_retweet"]
    res = res.loc[~drop]
    res["text"] = res["text"].str.lower()
    res["text"] = res["text"].str.slice(0, MAX_LEN)
    res["text"] += STOP_CHAR
    res = res["text"].drop_duplicates()
    return res


def compute_encoding(proc, min_nobs=100):
    chars = list("".join(proc))
    counts = collections.Counter(chars)
    keep = [k for k, v in counts.items() if v > min_nobs]
    encoding = dict(zip(keep, range(1, len(keep) + 1)))
    encoding = collections.defaultdict(lambda: 0, encoding)
    encoding[PAD_CHAR] = len(encoding)
    return encoding


def compute_decoding(encoding):
    return {v: k for k, v in encoding.items()}


def encode_data(proc, encoding):
    res = []
    pad_encode = [encoding[PAD_CHAR]]
    for tweet in tqdm(proc):
        encoded = np.array(
            [encoding[c] for c in tweet] + (MAX_LEN + 1 - len(tweet))*pad_encode
        )
        res.append(encoded)
    res = np.array(res)
    return res
