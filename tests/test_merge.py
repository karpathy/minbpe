import pytest

from minbpe.base import merge


def test_no_replacements():
    assert merge([1, 2, 3], (2, 1), 4) == [1, 2, 3]

def test_single_replacement():
    assert merge([1, 2], (1, 2), 4) == [4]
    assert merge([1, 2, 3], (1, 2), 4) == [4, 3]
    assert merge([0, 1, 2], (1, 2), 4) == [0, 4]
    assert merge([0, 1, 2, 3], (1, 2), 4) == [0, 4, 3]

def test_multiple_replacements():
    assert merge([1, 2, 3, 1, 2], (1, 2), 4) == [4, 3, 4]
    assert merge([1, 1, 2, 1, 2], (1, 2), 4) == [1, 4, 4]

def test_repeated_tokens():
    assert merge([1, 2, 1, 1, 2], (1, 1), 4) == [1, 2, 4, 2]
    assert merge([1, 1, 2, 1, 1, 1, 2], (1, 1), 4) == [4, 2, 4, 1, 2]

if __name__ == "__main__":
    pytest.main()
