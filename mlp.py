import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words= open('names.txt', 'r').read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
print(itos)

block_size=3 #context length: how many characters do we take to predict the next one? 
X=[] #input 
Y=[] #label
for w in words:
    
    context=[0]*block_size
    for ch in w+'.':
        ix=stoi[ch]
        X.append(context)
        Y.append(ix)
        
        context=context[1:]+[ix]
X=torch.tensor(X)
Y=torch.tensor(Y)
print(X.shape) #(32,3)
print(Y.shape) #(32)



# C=torch.randn((27,2)) #character embedding matrix
# emb=C[X] #embed the characters into vectors
# print(emb.shape)
# W1=torch.randn((6,100)) #weights for the hidden layer
# b1=torch.randn(100) #bias for the hidden layer
# h= emb.view(-1,6) @ W1 + b1 #hidden layer
# print(h.shape)
# W2=torch.randn((100,27)) #weights for the output layer
# b2=torch.randn(27) #bias for the output layer

# logits= h@W2 + b2
# print(logits.shape)

# # counts= logits.exp()
# # probs= counts/counts.sum(1,keepdim=True)
# # print(probs.shape)
# # loss=-probs[torch.arange(32), Y].log().mean()

# loss=F.cross_entropy(logits,Y) 
# print(loss)


g=torch.Generator().manual_seed(2147483647)
C=torch.randn((27,2),generator=g) #character embedding matrix
W1=torch.randn((6,100),generator=g) #weights for the hidden layer
b1=torch.randn(100,generator=g) #bias for the hidden layer
W2=torch.randn((100,27),generator=g) #weights for the output layer
b2=torch.randn(27,generator=g) #bias for the output layer
parameters=[C,W1,b1,W2,b2]
for p in parameters:
    p.requires_grad=True


num_param=sum(p.nelement() for p in parameters)
print(f'{num_param:,} parameters')


lre=torch.linspace(-3,0,1000)
lrs=10**lre 

lri=[]
lossi=[]
for k in range(1000):
    #minibatch
    ix=torch.randint(0,X.shape[0],(32,)) #generate a random batch of 32 indices

    #forward pass
    emb=C[X[ix]] #embed the characters into vectors #(32,3,2) 
    h= torch.tanh(emb.view(-1,6) @ W1 + b1) #hidden layer #(32,100)   
    logits= h@W2 + b2 #(32,27)
    loss=F.cross_entropy(logits,Y[ix])
    print(loss.item())  

    #backward pass
    for p in parameters:
        p.grad=None
    loss.backward() 

    #update
    lr=0.1
    
    for p in parameters:
        p.data += -lr * p.grad 
    
    # lri.append(lre[k].item())
    # lossi.append(loss.item())
# plt.plot(lri,lossi) 
# plt.show()

#training set, validation set, test set
#80%, 10%, 10%

# def build_dataset(words):
#     block_size=3 #context length: how many characters do we take to predict the next one? 
#     X=[] #input 
#     Y=[] #label
#     for w in words:
    
#         context=[0]*block_size
#         for ch in w+'.':
#             ix=stoi[ch]
#             X.append(context)
#             Y.append(ix)
        
#             context=context[1:]+[ix]
#     X=torch.tensor(X)
#     Y=torch.tensor(Y)
#     return X,Y

# import random
# random.seed(2147483647)
# random.shuffle(words)
# n1=int(0.8*len(words))
# n2=int(0.9*len(words))
# Xtr,Ytr=build_dataset(words[:n1])
# Xdev,Ydev=build_dataset(words[n1:n2])
# Xte,Yte=build_dataset(words[n2:])
# print(Xtr.shape,Ytr.shape)
# print(Xdev.shape,Ydev.shape)        
# print(Xte.shape,Yte.shape)

#sampling from the model

g=torch.Generator().manual_seed(2147483647)
for _ in range(10):
    out=[]
    context=[0]*block_size
    while True:
        emb=C[torch.tensor([context])] #embed the characters into vectors #(1,3,2) 
        h= torch.tanh(emb.view(-1,6) @ W1 + b1) #hidden layer #(1,100)   
        logits= h@W2 + b2 #(1,27)
        probs=F.softmax(logits,dim=1)
        ix=torch.multinomial(probs,num_samples=1,generator=g).item()
        out.append(itos[ix])
        context=context[1:]+[ix]
        if ix==0:
            break
    print(''.join(out))