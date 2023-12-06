from constants import NUM_BPP, NUM_REACTIVITIES
import itertools
import numpy as np


class Encoder:
    def __init__(self) -> None:
        # one hot matrices
        self.simple_tokens = {
            "A": 1,
            "U": 2,
            "C": 3,
            "G": 4,
        }

        self._create_tokenizer()

    def _create_tokenizer(self) -> None:
        # create the basic options for what the base pairs and mfe can look like
        bases = {"A", "U", "C", "G"}
        mfes = {"(", ".", ")"}

        expanded_mfe = list(
            map(
                lambda tup: self._flatten_tuple(tup),
                itertools.product(mfes, repeat=NUM_BPP),
            )
        )
        self.tokens = {
            key: val
            for key, val in zip(
                map(
                    lambda tup: self._flatten_tuple(tup),
                    itertools.product(bases, expanded_mfe),
                ),
                # reserve 0 for padding value
                range(1, len(expanded_mfe) * len(bases) + 1),  # |AxB| = |A||B|
            )
        }

    def _flatten_tuple(self, tup):
        """
        Helper function that flattens a tuple
        """
        ret = tup[0]
        for i in range(1, len(tup)):
            ret += tup[i]
        return ret

    def tokenize(self, seq: str, *mfes: str) -> np.ndarray:
        # check that everything is the same length
        length = len(seq)
        for item in mfes:
            assert len(item) == length, "Sequence and MFE must be same length"

        # tokenize
        ret = np.zeros((NUM_REACTIVITIES,), dtype=np.int16)
        ret[:length] = np.array(
            list(
                map(lambda seq_, *mfes_: self.tokens[seq_ + "".join(mfes_)], seq, *mfes)
            )
        )
        return ret

    def simple_tokenize(self, seq: str) -> np.ndarray:
        ret = np.zeros((NUM_REACTIVITIES,), dtype=np.int16)
        ret[: len(seq)] = np.array(list(map(lambda nt: self.simple_tokens[nt], seq)))
        return ret

    def num_tokens(self) -> int:
        return len(self.tokens) + 1
