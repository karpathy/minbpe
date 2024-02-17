import pytest
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
        "!@#$%^&*()_+{}[];:'\",.<>?/`~"
    ])
    def test_encode_decode_roundtrip(self, tokenizer, text):
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert text == decoded
    
    # reference test to add more tests in the future
    # taken from bpe_base.py
    def test_bpe_basic(self):
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
        tokenizer = BasicTokenizer()
        tokenizer.train(text, 256 + 3)
        ids = tokenizer.encode(text)
        assert ids == [258, 100, 258, 97, 99]
        assert tokenizer.decode(tokenizer.encode(text)) == text
        
if __name__ == "__main__":
    pytest.main()
