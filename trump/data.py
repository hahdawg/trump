import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

from trump import config

STOP_CHAR = "\u2665"
PAD_CHAR = "\u2666"
MAX_LEN = 280


def load_data():
    res = pd.read_csv(config.data_path)
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


def compute_encoder(proc, min_nobs=100):
    chars = list("".join(proc))
    counts = collections.Counter(chars)
    keep = [k for k, v in counts.items() if v > min_nobs]
    encoder = dict(zip(keep, range(1, len(keep) + 1)))
    encoder = collections.defaultdict(lambda: 0, encoder)
    encoder[PAD_CHAR] = len(encoder)
    return encoder


def compute_decoder(encoder):
    decoder = {v: k for k, v in encoder.items()}
    decoder[0] = "<UNK>"
    return decoder


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


def main():
    raw = load_data()
    proc = process_data(raw)
    encoder = compute_encoder(proc)
    decoder = compute_decoder(encoder)
    encoded = encode_data(proc, encoder)
    return encoded, encoder, decoder
