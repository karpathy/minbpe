import sys
from collections import Counter

import pytest
from minbpe.base import get_stats, merge

def test_get_stats():
    ids = [1, 2, 3, 1, 2]
    counts = get_stats(ids)
    assert counts == Counter({(1, 2): 2, (2, 3): 1, (3, 1): 1})

def test_merge():
    tokens = [1, 2, 3, 1, 2]
    pair = (1, 2)
    merged_token = 4
    new_tokens = merge(tokens, pair, merged_token)
    assert new_tokens == [4, 3, 4]

if __name__ == "__main__":
    pytest.main(sys.argv[1:] + ["--verbose", "--verbose", "--verbose"])