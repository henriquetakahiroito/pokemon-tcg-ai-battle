"""BC step 2: train an option-scorer to imitate top-player moves; report per-decision
top-1 accuracy (does argmax match what the human chose?) vs the random baseline.
Export weights to numpy for submission-safe inference."""
import sys; sys.path.insert(0,'.'); sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, torch, torch.nn as nn
d=np.load('_bc_data.npz'); X=d['X']; Y=d['Y']; G=d['G']
ndec=int(G.max())+1
rng=np.random.default_rng(0); perm=rng.permutation(ndec)
test_dec=set(perm[:ndec//5].tolist())            # 20% of decisions held out
test_mask=np.array([g in test_dec for g in G])
Xtr,Ytr=X[~test_mask],Y[~test_mask]; Xte,Yte,Gte=X[test_mask],Y[test_mask],G[test_mask]
Xt=torch.tensor(Xtr); Yt=torch.tensor(Ytr)
mu=Xt.mean(0,keepdim=True); sd=Xt.std(0,keepdim=True)+1e-6
net=nn.Sequential(nn.Linear(X.shape[1],64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(Ytr==0).sum()/(Ytr==1).sum()]))
Xn=(Xt-mu)/sd
for ep in range(40):
    net.train(); idx=torch.randperm(len(Xn))
    for i in range(0,len(Xn),4096):
        b=idx[i:i+4096]; opt.zero_grad()
        l=lossf(net(Xn[b]).squeeze(1),Yt[b]); l.backward(); opt.step()
# eval: per-decision top-1 accuracy
net.eval()
with torch.no_grad():
    sc=net(((torch.tensor(Xte)-mu)/sd)).squeeze(1).numpy()
from collections import defaultdict
groups=defaultdict(list)
for k in range(len(Gte)): groups[Gte[k]].append(k)
hit=tot=0; rand=0.0
for g,ks in groups.items():
    ks=np.array(ks); chosen=ks[np.argmax(Yte[ks])]
    pred=ks[np.argmax(sc[ks])]
    hit+= (pred==chosen); tot+=1; rand+=1.0/len(ks)
print(f'held-out decisions: {tot}')
print(f'  BC model top-1 accuracy : {hit/tot:.1%}')
print(f'  random baseline         : {rand/tot:.1%}')
# export to numpy
w={}
for i,layer in enumerate([m for m in net if isinstance(m,nn.Linear)]):
    w[f'W{i}']=layer.weight.detach().numpy(); w[f'b{i}']=layer.bias.detach().numpy()
w['mu']=mu.numpy(); w['sd']=sd.numpy()
np.savez('_bc_weights.npz', **w)
print('exported _bc_weights.npz (numpy inference, submission-safe)')
