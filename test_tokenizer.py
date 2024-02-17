import pytest
import tiktoken
import os

from bpe_basic import BasicTokenizer
from bpe_gpt4 import GPT4Tokenizer
from bpe_regex import RegexTokenizer


@pytest.fixture(params=[BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
def tokenizer(request):
    return request.param()

class TestTokenizer:
    """
    Test the encode() and decode() methods of the Tokenizer.
    Few Cases:
    - empty string
    - multi-lingual string
    - a longer piece of text (from a file)
    - a piece of text with special characters
    """
    @pytest.mark.parametrize("text", [
        "",
        "hello world!!!? (안녕하세요!) lol123 😉",
        open("taylorswift.txt", "r", encoding="utf-8").read(), 
        # disclaimer: disable the above line during debugging as it prints a lot
        "!@#$%^&*()_+{}[];:'\",.<>?/`~"
    ])
    def test_encode_decode_roundtrip(self, tokenizer, text):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        if isinstance(tokenizer, GPT4Tokenizer):
            tiktoken_enc = tiktoken.get_encoding("cl100k_base")
            assert encoded == tiktoken_enc.encode(text), f"{tokenizer}encoding does not match tiktoken"
            
        assert text == decoded
    
    # reference test to add more tests in the future
    # taken from bpe_base.py
    def test_tokenizer_wikipedia_example(self, tokenizer):
        """
        Quick unit test, following along the Wikipedia example:
        https://en.wikipedia.org/wiki/Byte_pair_encoding

        According to Wikipedia, running bpe on the the input string:
        "aaabdaaabac"

        for 3 merges will result in string:
        "XdXac"

        where:
        X=ZY
        Y=ab
        Z=aa

        Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
        so Z will be 256, Y will be 257, X will be 258.

        So we expect the output list of ids to be [258, 100, 258, 97, 99]
        """
        text = "aaabdaaabac"
        if not isinstance(tokenizer, GPT4Tokenizer):
            tokenizer.train(text, 256 + 3)
            ids = tokenizer.encode(text)
            assert ids == [258, 100, 258, 97, 99]
            assert tokenizer.decode(tokenizer.encode(text)) == text
        
    def test_tokenizer_model_save_load(self):
        # take a bit more complex piece of text and train the tokenizer, chosen at random
        text = """
        The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
        Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
        The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
        In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology, llamas will return to the water springs and ponds where they come from at the end of time.[6]
        """.strip()
        
        tokenizer = RegexTokenizer()
        
        # do 64 merges
        tokenizer.train(text, 256 + 64)

        # verify that decode(encode(x)) == x
        assert tokenizer.decode(tokenizer.encode(text)) == text

        # verify that save/load work as expected
        ids = tokenizer.encode(text)

        # save the tokenizer
        tokenizer.save("toy")
        # re-load the tokenizer
        tokenizer = RegexTokenizer()
        tokenizer.load("toy.model")

        # verify that decode(encode(x)) == x
        assert tokenizer.decode(ids) == text
        assert tokenizer.decode(tokenizer.encode(text)) == text
        assert tokenizer.encode(text) == ids
        
        # delete the saved file artifacts after the test
        for file in ["toy.model", "toy.vocab"]:
            os.remove(file)
        os.system("rm -rf __pycache__")


if __name__ == "__main__":
    pytest.main()
