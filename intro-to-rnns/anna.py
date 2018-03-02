import time
from collections import namedtuple

import numpy as np
import tensorflow as tf

with open('anna.txt', 'r') as f:
    text=f.read()
vocab = sorted(set(text))
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

text[:100]

print(encoded[:100])

len(vocab)
