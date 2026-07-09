# define.py


PII = tuple[int, int]


class pair_hash:
    def __call__(self, pair):
        hash1 = hash(pair[0])
        hash2 = hash(pair[1])
        return hash1 ^ (hash2 << 1)
