import torch
import torch.nn as nn
import torch.nn.functional as F
from mp_deterministic import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag
from GAT import GAT

class ONGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, params):
        super(ONGNNConv, self).__init__('mean')
        self.params = params
        self.tm_net = tm_net
        self.tm_norm = tm_norm
        self.beta = nn.Parameter(0.0 * torch.ones(size=(1, 1)))
        self.gat_net = GAT(params['hidden_channel'], params['hidden_channel'], params['dropout_rate3'], 0.2, params['heads'], False)

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.params['add_self_loops']==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.params['add_self_loops']==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        edge_index = edge_index.to_dense()
        m = self.gat_net(x, edge_index)
        if self.params['tm']==True:
            if self.params['simple_gating']==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.params['diff_or']==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)

            beta = torch.sigmoid(self.beta)
            x = x * tm_signal
            m = m * (1 - tm_signal)
            out = x * beta + m * (1 - beta)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw