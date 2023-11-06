import torch
from gpt import Decoder
from tokenizer import Tokenizer
from tqdm import tqdm

#Parameters
heads_nb=8
d_k=16
d_embedding=128
blocks_nb=8
max_window_size=20
b_size=16
epochs_nb=5000

with open("training.text",'r') as file:
    text=file.read()

vocab=sorted(list(set(text)))
vocab_size=len(vocab)

tokenizer=Tokenizer(vocab)
data=tokenizer.encode(text)

def batching(data, b_size):
    random_idx=torch.randint(0, len(data)-max_window_size, (b_size,))
    batch_x=torch.stack([data[idx:idx+max_window_size] for idx in random_idx])
    batch_y=torch.stack([data[idx+1:idx+max_window_size+1] for idx in random_idx]) #(b_size, max_window_size)
    return batch_x,batch_y

#Initializing model
model=Decoder(heads_nb, d_k, d_embedding, blocks_nb, max_window_size, vocab_size)
#Initializing optimizer
optimizer = torch.optim.AdamW (model.parameters(), lr=1e-3)
#Initializing loss function
loss_fn=torch.nn.CrossEntropyLoss()

#Training
for epoch in tqdm(range(epochs_nb)):
    batch_x, batch_y=batching(data, b_size)
    optimizer.zero_grad()
    y_pred=model(batch_x)
    #reshaping y_pred and batch_y for the loss function
    b_y,t_y, v_y=y_pred.shape
    y_pred=y_pred.reshape(b_y*t_y, v_y)
    batch_y=batch_y.reshape(b_y*t_y)
    loss=loss_fn(y_pred, batch_y)
    loss.backward()
    optimizer.step()
    print(loss.item())

#Generating
context="Une souris"
coded_context=tokenizer.encode(context)
completed_text=model.generate(coded_context, 20)
print(tokenizer.decode((completed_text).tolist()))

