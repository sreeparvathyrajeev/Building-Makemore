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

# trained model parameters loaded 
checkpoint = torch.load('model.pt')
C, W1, b1, W2, b2 = checkpoint['C'], checkpoint['W1'], checkpoint['b1'], checkpoint['W2'], checkpoint['b2']

@torch.no_grad()
def split_loss(split):
    if split == 'train':
        X, Y = Xtr, Ytr
    elif split == 'dev':
        X, Y = Xdev, Ydev
    elif split == 'test':
        X, Y = Xte, Yte
    else:
        raise ValueError("Invalid split name")
    
    # forward pass
    emb=C[X] # (N, block_size, hidden_size)
    embcat=emb.view(emb.shape[0],-1) #(N, block_size * hidden_size)
    hpreact = embcat @ W1 + b1  # (N, hidden_size)
    h = torch.tanh(hpreact)  # (N, hidden_size)
    logits = h @ W2 + b2  # (N, vocab_size)
    
    loss = F.cross_entropy(logits, Y)
    return loss.item()

print(f"train loss: {split_loss('train'):.4f}")
print(f"dev loss: {split_loss('dev'):.4f}")

