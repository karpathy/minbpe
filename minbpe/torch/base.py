import torch
from torch import Tensor

def merge_torch(ids: Tensor, pair: Tensor, idx: int):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """

    # create a mask for the first element i of every matching pair (i, j)
    pairs = torch.stack((ids[:-1], ids[1:]), dim=1)
    is_pair = (pairs == pair).all(axis=1)
    false_tensor = torch.tensor([False], dtype=torch.bool, device=ids.device)
    is_pair_i = torch.cat((is_pair, false_tensor))

    # create a mask for the second element j of every matching pair (i, j)
    is_pair_j = is_pair_i.roll(1)

    # handle overlapping pairs for repeated tokens
    while True:
        is_overlap = (is_pair_i & is_pair_j).any()
        if not is_overlap:
            break # no overlapping pairs

        # remove first overlapping pairs in repeated sequences
        is_first = (is_pair_i & is_pair_j).int().diff() == 1
        is_first = torch.cat((false_tensor, is_first))
        is_pair_i &= ~is_first
        is_pair_j = is_pair_i.roll(1)

    # change the first element i of every matching pair (i, j) to the new token
    ids[is_pair_i] = idx

    # remove the second element j of every matching pair (i, j)
    ids = ids[~is_pair_j]
    return ids