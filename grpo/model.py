"""RWKV-7 BF16 equations from RWKV-LM-V7/src/model.py, without Lightning.

Recurrent state layout is [layer,batch,head,value,key]. Shift states are runtime
activations, initialized to zero, never trainable. Training retains their graph.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .kernel import wkv


class Backbone(nn.Module):
    def __init__(self, path, device='cuda'):
        super().__init__()
        weights = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self.n_layer = max(int(k.split('.')[1]) for k in weights if k.startswith('blocks.')) + 1
        self.n_embd = weights['emb.weight'].shape[1]
        self.n_head, self.head_size = weights['blocks.0.att.r_k'].shape
        if self.head_size != 64:
            raise ValueError('Only standard RWKV-7 head_size=64 is supported')
        self.weights = nn.ParameterDict({k.replace('.', '__'): nn.Parameter(v.to(device=device, dtype=torch.bfloat16), requires_grad=False)
                                         for k,v in weights.items()})
        self.eval()

    def weight(self, name):
        return self.weights[name.replace('.', '__')]

    def norm(self, x, name):
        return F.layer_norm(x, (self.n_embd,), self.weight(name+'.weight'), self.weight(name+'.bias'))

    def forward(self, ids, states, shifts=None, last_only=False):
        B, T = ids.shape
        C, H = self.n_embd, self.n_head
        x = F.embedding(ids, self.weight('emb.weight'))
        if shifts is None:
            shifts = x.new_zeros(self.n_layer, 2, B, C)
        final_states, final_shifts = [], []
        v_first = None
        for i in range(self.n_layer):
            p = f'blocks.{i}.'
            if i == 0:
                x = self.norm(x, p+'ln0')
            z = self.norm(x, p+'ln1')
            xx = torch.cat((shifts[i,0,:,None,:], z[:,:-1]), dim=1) - z
            get = lambda key: self.weight(p+'att.'+key)
            xr,xw,xk,xv,xa,xg = [z + xx*get('x_'+c) for c in 'rwkvag']
            r = F.linear(xr, get('receptance.weight'))
            w = get('w0') + torch.tanh(xw @ get('w1')) @ get('w2')
            k = F.linear(xk, get('key.weight'))
            v = F.linear(xv, get('value.weight'))
            if i == 0:
                v_first = v
            else:
                v = v + (v_first-v)*torch.sigmoid(get('v0') + (xv@get('v1'))@get('v2'))
            a = torch.sigmoid(get('a0') + (xa@get('a1'))@get('a2'))
            g = torch.sigmoid(xg@get('g1'))@get('g2')
            kk = F.normalize((k*get('k_k')).reshape(B,T,H,64), dim=-1).reshape(B,T,C)
            k = k*(1+(a-1)*get('k_a'))
            y, s = wkv(states[i], r,w,k,v,-kk,kk*a)
            y = F.group_norm(y.reshape(B*T,C), H, get('ln_x.weight'), get('ln_x.bias'), eps=64e-5).reshape(B,T,C)
            y = y + ((r.reshape(B,T,H,64)*k.reshape(B,T,H,64)*get('r_k')).sum(-1,keepdim=True)*v.reshape(B,T,H,64)).reshape(B,T,C)
            x = x + F.linear(y*g, get('output.weight'))
            f = self.norm(x, p+'ln2')
            xx = torch.cat((shifts[i,1,:,None,:], f[:,:-1]), dim=1)-f
            kf = f + xx*self.weight(p+'ffn.x_k')
            x = x + F.linear(F.relu(F.linear(kf,self.weight(p+'ffn.key.weight'))).square(),self.weight(p+'ffn.value.weight'))
            final_states.append(s)
            final_shifts.append(torch.stack((z[:,-1],f[:,-1])))
        if last_only:
            x = x[:,-1:]
        logits = F.linear(self.norm(x,'ln_out'),self.weight('head.weight'))
        return logits, torch.stack(final_states), torch.stack(final_shifts)


class StatePolicy(nn.Module):
    def __init__(self, backbone, initial_state):
        super().__init__()
        self.backbone = backbone
        shape = (backbone.n_layer, backbone.n_head, 64, 64)
        if tuple(initial_state.shape) != shape or not torch.isfinite(initial_state).all():
            raise ValueError(f'Invalid initial state: expected {shape}')
        self.initial_state = nn.Parameter(initial_state.to(device=backbone.weight('emb.weight').device, dtype=torch.float32))

    def start(self, batch):
        return self.initial_state[:,None].expand(-1,batch,-1,-1,-1).contiguous()

    def forward(self, ids, targets=None, chunk_size=128):
        # Chunking bounds vocabulary-logit memory; state and shifts remain attached.
        states, shifts = self.start(ids.shape[0]), None
        outputs = []
        for start in range(0,ids.shape[1],chunk_size):
            logits,states,shifts = self.backbone(ids[:,start:start+chunk_size],states,shifts)
            if targets is None:
                outputs.append(logits)
            else:
                logp = F.log_softmax(logits.float(),dim=-1)
                outputs.append(logp.gather(-1,targets[:,start:start+chunk_size,None]).squeeze(-1))
        return torch.cat(outputs,dim=1)
