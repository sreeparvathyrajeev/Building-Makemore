import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

device='cuda' 
print(torch.cuda.is_available())  # should print True
print(torch.cuda.get_device_name(0))  # should show your GPU name

words= open('names.txt', 'r').read().splitlines()
chars=sorted(list(set(''.join(words))))
stoi={s:i+1 for i,s in enumerate(chars)}
stoi['.']=0
itos={i:s for s,i in stoi.items()}
vocab_size=len(itos)
print(itos)





#training set, validation set, test set
#80%, 10%, 10%
block_size=3 #context length: how many characters do we take to predict the next one? 
def build_dataset(words):
    
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
    return X,Y

import random
random.seed(2147483647)
random.shuffle(words)
n1=int(0.8*len(words))
n2=int(0.9*len(words))
Xtr,Ytr=build_dataset(words[:n1])
Xdev,Ydev=build_dataset(words[n1:n2])
Xte,Yte=build_dataset(words[n2:])
Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xdev, Ydev = Xdev.to(device), Ydev.to(device)
Xte, Yte = Xte.to(device), Yte.to(device)



torch.save({
    'Xtr': Xtr, 'Ytr': Ytr,
    'Xdev': Xdev, 'Ydev': Ydev,
    'Xte': Xte, 'Yte': Yte
}, 'dataset.pt')

print(Xtr.shape,Ytr.shape)
print(Xdev.shape,Ydev.shape)        
print(Xte.shape,Yte.shape)
