import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(torch.nn.Module):
    
    def __init__(self, input_dim, mlp_dim_list, dropout, bn=False, activation=None, output_layer=True):
        super().__init__()
        layers = list()
        for mlp_dim in mlp_dim_list:
            layers.append(nn.Linear(input_dim, mlp_dim))
            if bn:
                layers.append(nn.BatchNorm1d(mlp_dim))
            if activation != None:
                layers.append(activation)
            layers.append(nn.Dropout(p=dropout))
            input_dim = mlp_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class Attention(torch.nn.Module):
    def __init__(self, att_input_dim, ModelSettings):
        super(Attention, self).__init__()
        hid_dim = att_input_dim

        self.att_func = ModelSettings['att_func']
        if self.att_func == 'dot':
            pass
        elif self.att_func == 'concat':
            self.w = nn.Linear(hid_dim * 2, hid_dim)
            self.v = nn.Linear(hid_dim, 1)
        elif self.att_func == 'add':
            self.w = nn.Linear(hid_dim, hid_dim)
            self.u = nn.Linear(hid_dim, hid_dim)
            self.v = nn.Linear(hid_dim, 1)
        elif self.att_func == 'linear':
            self.w = nn.Linear(hid_dim, hid_dim)
        elif self.att_func == 'din_activate':
            hid_dim_list = eval(ModelSettings['din_att_dim_list'])
            # self.mlp = MultiLayerPerceptron(hid_dim*3, hid_dim_list, dropout=0, activation=nn.Sigmoid())
            self.mlp = MultiLayerPerceptron(hid_dim*4, hid_dim_list, dropout=0, activation=nn.Sigmoid())
        self._init_weights()

    def forward(self, query, seq, seq_lens=None, given_mask=None):
        if self.att_func == 'dot':
            query = query.squeeze().unsqueeze(1)
            a = self.mask_softmax(torch.bmm(seq, query.permute(0, 2, 1)), seq_lens, 1)
            return a
        elif self.att_func == 'concat': 
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze().unsqueeze(1)
            a = torch.cat([seq, query.repeat([1, seq_len, 1])], 2).reshape([seq_len * batch_size, -1])
            a = F.relu(self.w(a))
            a = F.relu(self.v(a))
            a = self.mask_softmax(a.reshape([batch_size, seq_len, 1]), seq_lens, 1)
            return a
        elif self.att_func == 'add':
            seq_len = len(seq[0])
            batch_size = len(seq)
            seq = self.w(seq.reshape([batch_size * seq_len, -1]))
            query = self.u(query).repeat([seq_len, 1])
            a = self.mask_softmax(self.v(F.tanh(seq + query)).reshape([batch_size, seq_len, 1]), seq_lens, 1)
            return a
        elif self.att_func == 'linear':
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze()
            query = self.w(query).unsqueeze(2)
            a = self.mask_softmax(torch.bmm(seq, query), seq_lens, 1)
            return a
        elif self.att_func == 'din_activate':
            seq_len = len(seq[0])
            batch_size = len(seq)
            query = query.squeeze().unsqueeze(1)
            query = query.repeat([1, seq_len, 1]) 
            din_activate = torch.cat([query, seq, query - seq, query * seq], dim=-1)
            # din_activate = torch.cat([query, seq, query * seq], dim=-1)
            din_activate = din_activate.view(batch_size * seq_len, -1)
            a = self.mlp(din_activate)
            a = a.view(batch_size, seq_len)
            # Scale
            dim = len(seq[2]) # self.hid_dim
            a = a/(dim ** 0.5)
            # Mask
            # a = self.mask_softmax(a, seq_lens, 1)
            if given_mask is None and seq_len is not None:
                mask = (torch.arange(seq_len, device=seq_lens.device).repeat(
                    batch_size, 1) < seq_lens.view(-1, 1))
            elif given_mask is not None:
                mask = given_mask
            else:
                raise ValueError('You should give a seq_len or given_mask in attention layer !!!!')
            a[~mask] = -np.inf
            # Activation
            a = F.softmax(a, dim=1)
            return a

    def _init_weights(self):
        if self.att_func == 'dot':
            pass
        elif self.att_func == 'concat':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.att_func == 'add':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.u.weight, a=-0.1, b=0.1)
            nn.init.uniform_(self.v.weight, a=-0.1, b=0.1)
        elif self.att_func == 'linear':
            nn.init.uniform_(self.w.weight, a=-0.1, b=0.1)
    
    def mask_softmax(self, seqs, seq_lens=None, dim=1):
        if seq_lens is None:
            res = F.softmax(seqs, dim=dim)
        else:
            max_len = len(seqs[0])
            batch_size = len(seqs)
            ones = seq_lens.new_ones(batch_size, max_len, device=seq_lens.device)
            range_tensor = ones.cumsum(dim=1)
            mask = (seq_lens.unsqueeze(1) >= range_tensor).long()
            mask = mask.float()
            mask = mask.unsqueeze(2)
            # masked_vector = seqs.masked_fill((1 - mask).byte(), -1e32)
            masked_vector = seqs.masked_fill((1 - mask).bool(), -1e32)
            res = F.softmax(masked_vector, dim=dim)
        return res
