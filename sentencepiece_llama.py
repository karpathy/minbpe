import sentencepiece as spm

class LLamaTokenizer:
    def __init__(self, model_path=None):
        self.sp = spm.SentencePieceProcessor()
        if model_path:
            self.load_model(model_path)

    def train(self, file_path, output_model_path, vocab_size=4846, symbols=None):
        if symbols is not None:
            with open("symbols.txt", "w", encoding="utf-8") as symbols_file:
                symbols_file.write("\n".join(symbols))

        spm.SentencePieceTrainer.train(
            input=file_path,
            model_prefix=output_model_path,
            vocab_size = vocab_size,
            user_defined_symbols="symbols.txt" if symbols else None
        )
        self.load_model(output_model_path + ".model")

    def load_model(self, model_path):
        self.sp.load(model_path)

    def encode(self, s):
        return self.sp.encode_as_ids(s)

    def decode(self, l):
        return self.sp.decode_ids(l)



if __name__ == "__main__":
    # usage:
    file_path = '/workspaces/minbpe/taylorswift.txt'
    output_model_path = '/workspaces/minbpe/taylorswift_tokenizer.model'

    # instantiate the tokenizer
    tokenizer = LLamaTokenizer()

    custom_symbols = ",,!,,,,#,$,&,',(,),+,',-,.,/,0,1,2,3,4,5,6,7,8,9,:,;,?,@,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,[,],a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,£,®,á,é,í,ñ,ö,–,—,•,™"
    # training
    tokenizer.train(file_path, output_model_path, symbols=custom_symbols)

    # testing
    input_text = "hello"
    encoded_text = tokenizer.encode(input_text)
    print("Encoded:", encoded_text)

    decoded_text = tokenizer.decode(encoded_text)
    print("Decoded:", decoded_text)