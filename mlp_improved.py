import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device='cuda' 
print(torch.cuda.is_available())  # should print True
print(torch.cuda.get_device_name(0))  # should show your GPU name






#training set, validation set, test set loaded from file
#80%, 10%, 10%
data = torch.load('dataset.pt')
Xtr, Ytr = data['Xtr'].to(device), data['Ytr'].to(device)
Xdev, Ydev = data['Xdev'].to(device), data['Ydev'].to(device)
Xte, Yte = data['Xte'].to(device), data['Yte'].to(device)



#model parameters
block_size=3
vocab_size=27
n_embed=10 #the size of the character embedding vectors
n_hidden=200 #the number of neurons in the hidden layer
g=torch.Generator().manual_seed(2147483647)
C=torch.randn((vocab_size,n_embed),generator=g).to(device) #character embedding matrix
W1=torch.randn((n_embed*block_size,n_hidden),generator=g).to(device) #weights for the hidden layer
b1=torch.randn(n_hidden,generator=g).to(device) #bias for the hidden layer
W2=torch.randn((n_hidden,vocab_size),generator=g).to(device)*0.01 #weights for the output layer
b2=torch.randn(vocab_size,generator=g).to(device)*0 #bias for the output layer
parameters=[C,W1,b1,W2,b2]
print(f"number of parameters: {sum(p.nelement() for p in parameters)}")
for p in parameters:
    p.requires_grad=True 




#training loop

max_steps=200000
batch_size=32
lossi=[]

for k in range(max_steps):

    #minibatch
    ix=torch.randint(0,Xtr.shape[0],(batch_size,)) #generate a random batch of 32 indices
    Xb=Xtr[ix]
    Yb=Ytr[ix]

    #forward pass
    emb=C[Xb] #embed the characters into vectors #(32,3,10)
    embcat=emb.view(emb.shape[0],-1) #(32,30)
    hpreact=embcat @ W1 +b1 #(32,200) 
    h= torch.tanh(hpreact) #hidden layer #(32,200)     
    logits= h@W2 + b2 #(32,27)
    loss=F.cross_entropy(logits,Yb)
    

    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward() 

    #update
    lr=0.1 if k<100000 else 0.01 #step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad 

    #track stats
    lossi.append(loss.log10().item())
    if k%10000==0:
        print(f"step {k:7d} / {max_steps:7d}: loss {loss.item():.4f}")
    
    break
plt.hist(h.view(-1).tolist(),50)
plt.show()
# plt.plot(lossi)
# plt.show()


# torch.save({
#     'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2
# }, 'model.pt')

