# match this
import os
import tiktoken
from minbpe import RegexTokenizer, GPT4Tokenizer
enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer

print("\nStep 3\n")
text = "hello world!!!? (안녕하세요!) lol123 😉"
print(f"\ninput text: {text}")
ids = enc.encode(text)
print(f"\nids: {ids}")
decoded_text = enc.decode(ids) # get the same text back
if decoded_text != text:
    print(f"\ndecoded_text not match input text, decoded_text: {decoded_text}")
else:
    print(f"\ndecoded_text match input text")

my_gpt4_tokenizer = GPT4Tokenizer()
my_gpt4_ids = my_gpt4_tokenizer.encode(text)
print(f"\nmy_gpt4_ids: {my_gpt4_ids}")
my_gpt4_decoded_text = my_gpt4_tokenizer.decode(my_gpt4_ids) # get the same text back
if my_gpt4_decoded_text != text:
    print(f"\nmy_gpt4_decoded_text not match input text, my_gpt4_decoded_text: {my_gpt4_decoded_text}")
else:
    print(f"\nmy_gpt4_decoded_text match input text")

my_regex_tokenizer = RegexTokenizer()
my_regex_tokenizer.load(model_file=os.path.join("models", "regex.model"))
my_ids = my_regex_tokenizer.encode(text)
print(f"\nmy_regex_ids (without gpt4's byte_shuffle): {my_ids}")
my_decoded_text = my_regex_tokenizer.decode(my_ids) # get the same text back
if my_decoded_text != text:
    print(f"\nmy_decoded_text not match input text, my_decoded_text: {my_decoded_text}")
else:
    print(f"\nmy_decoded_text match input text")


print("\n\nStep 4")
enc = tiktoken.get_encoding("cl100k_base") # this is the GPT-4 tokenizer
ids = enc.encode("<|endoftext|>hello world", allowed_special="all")
# ids = enc.encode("<|endoftext|>hello world") # this will raise an error: ValueError: Encountered text corresponding to disallowed special token '<|endoftext|>'.
print(f"\nids with special tokens: {ids}")
