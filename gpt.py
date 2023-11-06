import torch
import torch.nn as nn
import torch.nn.functional as F

#One Head Attention
class OneHeadAttention(nn.Module):
    def __init__(self, d_k, d_embedding, masked): #dimension of the embedded vectors = number of channels
        super().__init__()
        self.d_k = d_k
        self.masked=masked
        self.Queries=nn.Linear(d_embedding, d_k, bias=False)
        self.Keys=nn.Linear(d_embedding, d_k, bias=False)
        self.Values=nn.Linear(d_embedding, d_k, bias=False)
    
    def forward(self,X): #X is matrix representing a group of embedded words (batch_size, words_nb, d_embedding)
        #shape of the following matrices : (batch_size, words_nb, d_k)
        q_x=self.Queries(X) 
        k_x=self.Keys(X) 
        v_x=self.Values(X) 
        #computing attention scores + normalization
        a_x=(q_x@k_x.transpose(-2,-1))*(self.d_k**-0.5) # (batch_size, words_nb, words_nb)
        #masking scores of the following words
        if self.masked:
            a_x=mask(a_x) #(batch_size, words_nb, words_nb)
        #taking the softmax
        a_x=F.softmax(a_x, dim=-1) #(batch_size, words_nb, words_nb)
        #multiplying by values
        y_x=a_x@v_x #(batch_size, words_nb, d_k)
        return y_x
    
#infinity tensor 
def mask(T):
    int_tensor=torch.tril(torch.ones(T.shape))
    inf_tensor=torch.zeros(T.shape)
    inf_tensor[int_tensor==0]=-float('inf')
    return T+inf_tensor

#Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, heads_nb, d_k, d_embedding, masked):
        super().__init__()
        self.heads_nb=heads_nb
        self.d_k=d_k
        self.Heads=nn.ModuleList([OneHeadAttention(d_k, d_embedding, masked) for _ in range(heads_nb)]) #we create the different heads
        self.W_proj=nn.Linear(heads_nb*d_k, d_embedding, bias=False)
    
    def forward(self,x):
        concat_tensor=torch.cat([h(x) for h in self.Heads], dim=-1) #(batch_size, words_nb, heads_nb*d_k)
        y_x=self.W_proj(concat_tensor) #(batch_size, words_nb, d_embedding) 
        return y_x

#Feed Forward
class FeedForward(nn.Module):
    def __init__(self, d_ff):
        super().__init__()
        self.d_ff=d_ff
        self.ff=nn.Sequential(nn.Linear(d_ff, d_ff*2), nn.ReLU(), nn.Linear(d_ff*2, d_ff), nn.Softmax())
    
    def forward(self,x):
        return self.ff(x)
    
#Decoder Block
class Block(nn.Module):
    def __init__(self, heads_nb, d_k, d_embedding):
        super().__init__()
        self.mmha=MultiHeadAttention(heads_nb, d_k, d_embedding, masked=True)
        self.ff=FeedForward(d_embedding)
        self.ln1=nn.LayerNorm(d_embedding)
        self.ln2=nn.LayerNorm(d_embedding)
    
    def forward(self,x):
        x=self.ln1(x+self.mmha(x))
        x=self.ln2(x+self.ff(x))
        return x
    

#Decoder
class Decoder(nn.Module):
    def __init__(self, heads_nb, d_k, d_embedding, blocks_nb, max_window_size, vocab_size):
        super().__init__()
        self.token_embed=nn.Embedding(vocab_size, d_embedding) #(batch_size, vocab_size, d_embedding)
        self.positional_embed=nn.Embedding(max_window_size, d_embedding) #(batch_size, max_window_size, d_embedding)
        self.blocks_nb=blocks_nb
        self.max_window_size=max_window_size
        self.Blocks=nn.Sequential(*[Block(heads_nb, d_k, d_embedding) for _ in range(blocks_nb)])
        self.ln_final=nn.Linear(d_embedding, vocab_size) #(batch_size, d_embedding, vocab_size)
    
    def forward(self,x): #x is a tensor composed of integers only depending on the batch size and the sequence length : (batch_size, tokens_nb)
        test=x.shape
        l_test=len(test)
        if l_test==1:
            tokens_nb=test[0]
        else:
            tokens_nb=test[1]
        if tokens_nb>self.max_window_size:
            raise Exception("Input sequence is too long")
        #encoding positions
        pos=torch.arange(tokens_nb) #(tokens_nb,)
        pos_embed=self.positional_embed(pos) #(tokens_nb, d_embedding)
        #embedding tokens
        tokens_emb=self.token_embed(x) #(batch_size, tokens_nb, d_embedding)
        #adding positional embedding
        x=tokens_emb+pos_embed #(batch_size, tokens_nb, d_embedding)
        #applying blocks
        x=self.Blocks(x) #(batch_size, tokens_nb, d_embedding)
        #linear projection
        y=self.ln_final(x) #(batch_size, tokens_nb, vocab_size)
        return y
    
    def generate(self, x, max_len): #x is a tensor representing the current context, x.shape=(,tokens_nb)
        tokens_nb=len(x)
        if tokens_nb>self.max_window_size:
            raise Exception("Input sequence is too long")
        for _ in range(max_len):
            x_window=x[-self.max_window_size:] 
            y_pred=self.forward(x_window) 
            #getting last word token
            last_token=y_pred[-1,:] #(,vocab_size)
            #predicting next word probas
            probas=F.softmax(last_token, dim=-1) #(,vocab_size)
            #selecting the next token
            next_token=torch.multinomial(probas, 1)
            #adding next token to x
            x=torch.cat([x, next_token], dim=-1) #(,tokens_nb+1)
        return x




        



        



