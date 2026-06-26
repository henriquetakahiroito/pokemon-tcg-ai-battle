"""BC v2 train: scalars + card embedding. Report per-decision top-1 accuracy and export numpy."""
import sys; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, torch, torch.nn as nn
d=np.load('_bc_data3.npz'); X=d['X']; C=d['C']; Y=d['Y']; G=d['G']; V=int(d['vocab_size'])
ndec=int(G.max())+1; rng=np.random.default_rng(0); perm=rng.permutation(ndec)
test=set(perm[:ndec//5].tolist()); tm=np.array([g in test for g in G])
mu=X[~tm].mean(0,keepdims=True); sd=X[~tm].std(0,keepdims=True)+1e-6
Xt=torch.tensor((X-mu)/sd); Ct=torch.tensor(C); Yt=torch.tensor(Y)
class Net(nn.Module):
    def __init__(s,nscal,V,e=16):
        super().__init__(); s.emb=nn.Embedding(V,e)
        s.mlp=nn.Sequential(nn.Linear(nscal+e,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(s,x,c): return s.mlp(torch.cat([x,s.emb(c)],1)).squeeze(1)
net=Net(X.shape[1],V); opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-5)
lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(Y[~tm]==0).sum()/(Y[~tm]==1).sum()]))
tr=np.where(~tm)[0]
for ep in range(40):
    net.train(); idx=torch.tensor(np.random.permutation(tr))
    for i in range(0,len(idx),4096):
        b=idx[i:i+4096]; opt.zero_grad()
        l=lossf(net(Xt[b],Ct[b]),Yt[b]); l.backward(); opt.step()
net.eval()
with torch.no_grad():
    te=np.where(tm)[0]; sc=net(Xt[te],Ct[te]).numpy()
from collections import defaultdict
grp=defaultdict(list)
for k,gi in zip(te,G[te]): grp[gi].append(k)
hit=tot=0; rand=0.0
sc_by={k:v for k,v in zip(te,sc)}
for gi,ks in grp.items():
    ks=np.array(ks); chosen=ks[np.argmax(Y[ks])]; pred=ks[np.argmax([sc_by[k] for k in ks])]
    hit+=(pred==chosen); tot+=1; rand+=1.0/len(ks)
print(f'held-out decisions {tot}')
print(f'  BC+embeddings top-1 : {hit/tot:.1%}')
print(f'  (no-embed baseline  : 36.8%)')
print(f'  random              : {rand/tot:.1%}')
w={'emb':net.emb.weight.detach().numpy(),'mu':mu,'sd':sd}
for i,l in enumerate([m for m in net.mlp if isinstance(m,nn.Linear)]):
    w[f'W{i}']=l.weight.detach().numpy(); w[f'b{i}']=l.bias.detach().numpy()
np.savez('_bc_weights3.npz', **w)
print('exported _bc_weights3.npz')
