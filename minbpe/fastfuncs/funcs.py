import ctypes
import os
import numpy as np
import time
from numpy.ctypeslib import ndpointer

# Define the structure for the result tuple
class Result(ctypes.Structure):
    _fields_ = [
        ('token_id', ctypes.c_int),
        ('first_id', ctypes.c_int),
        ('second_id', ctypes.c_int),
        ('token_list_len', ctypes.c_int),
        ('token_list', ctypes.POINTER(ctypes.c_int))
    ]

class TokenizeResult(ctypes.Structure):
    _fields_ = [
        ('result', ctypes.POINTER(ctypes.c_uint32)),
        ('length', ctypes.c_int)
    ]

# Load the DLL
script_dir = os.path.dirname(os.path.abspath(__file__))

if os.name == 'nt':  # Windows
    lib_extension = ".dll"
elif os.name == 'posix':  # Linux
    lib_extension = ".so"
else:
    raise OSError("Unsupported operating system")

dll_path = os.path.join(script_dir, f"fastFuncs{lib_extension}")
funcs = ctypes.CDLL(dll_path)

# Define the input and output types of the function
funcs.train.restype = ctypes.POINTER(Result)
funcs.train.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]

funcs.tokenize.restype = TokenizeResult
funcs.tokenize.argtypes = [ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), ctypes.c_int, 
                           ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"), ctypes.c_int, 
                           ctypes.POINTER(ctypes.c_int64), ctypes.c_int,
                           ctypes.c_int, ctypes.c_int]

def trainFast(ids, num_tokens, init_tokens=256):
    """
    trains on ids which are already separated 
    """
    # Call the c++ function:
    results_ptr = funcs.train(ids, len(ids), num_tokens, init_tokens)
    # Convert the results to a Python list of tuples:
    results = []
    if results_ptr[0].token_id == -1:
        raise RuntimeError("Too many tokens:( Decrease number of tokens or use more training data")
    for i in range(num_tokens):
        result = results_ptr[i]
        tok_len = result.token_list_len
        tokens = [result.token_list[j] for j in range(tok_len)]
        results.append((result.token_id, result.first_id, result.second_id, tokens))
    vocab = {}
    merges = {}
    for el in results:
        if el[0] >= init_tokens:
            merges[(el[1], el[2])] = el[0]
        vocab[el[0]] = bytes(el[3])
    return merges, vocab

def tokenizeFast(ids, split_indices, merges, init_tokens, threads=4):
    vocab_size = len(merges)+init_tokens
    merges_l = [pair[0] * vocab_size + pair[1] for pair in merges]
    merges_np = np.array(merges_l, dtype=np.int64)
    merges_arr = merges_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    # split_indices_arr = split_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    # ids_arr = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    results_ptr = funcs.tokenize(
        ids, len(ids), split_indices, len(split_indices),
        merges_arr, len(merges), init_tokens, threads
    )

    tokenized_text_size = results_ptr.length
    tokenized_text_buffer = (ctypes.c_uint32 * tokenized_text_size)()
    ctypes.memmove(tokenized_text_buffer, results_ptr.result, ctypes.sizeof(ctypes.c_uint32) * tokenized_text_size)
    tokenized_text = list(tokenized_text_buffer)
    return tokenized_text
