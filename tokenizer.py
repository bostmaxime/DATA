import torch

class Tokenizer:
    def __init__(self, vocab): #vocab is a list of strings
        self.vocab=vocab
        self.vocab_size=len(vocab)
        self.vocab2idx={v:i for i,v in enumerate(vocab)}
        self.idx2vocab={i:v for i,v in enumerate(vocab)}
    
    def encode(self, sentence):
        return torch.tensor([self.vocab2idx[w] for w in sentence],dtype=torch.long)
    
    def decode(self, encoded_sentence):
        return " ".join([self.idx2vocab[i] for i in encoded_sentence])
    

