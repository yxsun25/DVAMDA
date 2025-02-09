import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.utils import (add_self_loops, negative_sampling,
                               remove_self_loops)
from sklearn.metrics import (average_precision_score, roc_auc_score,
                           accuracy_score, precision_score, recall_score,
                           f1_score, matthews_corrcoef)
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, degree
import numpy as np
from torch_geometric.nn import SAGEConv

sc = 0.8
MAX_LOGSTD = 10

class VGAEEncoder1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEEncoder1, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels)
        self.linear2 = nn.Linear(in_channels, out_channels)
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index):
        x_ = self.linear1(x)
        x_ = self.propagate(x_, edge_index)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * sc
        x = self.propagate(x, edge_index)
        return x, x_

   
class VGAEEncoder2(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEEncoder2, self).__init__()
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        x_ = self.linear1(x)

        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1) * sc
        return x, x_
    
class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels,out_channels,num_layers=3 ):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers


        self.convs = nn.ModuleList()
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        ), train_eps=True))


        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU()
            ), train_eps=True))


        self.final_lin = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.final_lin.reset_parameters()

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = conv(x, edge_index)


        x = self.final_lin(x)

        return x

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, heads=4, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.dropout = dropout


        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))


        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))


        self.convs.append(GCNConv(hidden_channels, out_channels))


        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x) if len(self.bns) > i else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class SAGEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5):
        super(SAGEEncoder, self).__init__()
        self.dropout = dropout


        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))


        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))


        self.convs.append(SAGEConv(hidden_channels, out_channels))


        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)])

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x) if len(self.bns) > i else x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class LPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, encoder_layer, num_layers,
                 dropout):
        super(LPDecoder, self).__init__()
        n_layer = encoder_layer * encoder_layer
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels * n_layer, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def cross_layer(self, x_1, x_2):
        bi_layer = torch.mul(x_1, x_2)
        return bi_layer

    def forward(self, h, edge):
        src_x = h[edge[0]]
        dst_x = h[edge[1]]
        x = self.cross_layer(src_x, dst_x)
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        #return torch.sigmoid(x)
        return x
           
def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss    

class DVA(nn.Module):
    def __init__(self, InputEncoder, VGAEEncoder1, VGAEEncoder2, predictor, device):
        super(DVA, self).__init__()
        self.InputEncoder = InputEncoder
        self.VGAEEncoder1 = VGAEEncoder1
        self.VGAEEncoder2 = VGAEEncoder2
                
        self.predictor = predictor

        DVA.reset_parameters(self)
        self.negative_sampler = negative_sampling
        self.device = device
        self.loss_fn = ce_loss

    def reset_parameters(self):
        reset(self.InputEncoder)
        reset(self.VGAEEncoder1)
        reset(self.VGAEEncoder2)
        reset(self.predictor)
   
    def VGAEEncode1(self, *args, **kwargs):
        """"""
        self.__mu1__, self.__logstd1__ = self.VGAEEncoder1(*args, **kwargs)
        self.__logstd1__ = self.__logstd1__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu1__, self.__logstd1__)
        return z
    
    def VGAEEncode2(self, *args, **kwargs):
        """"""
        self.__mu2__, self.__logstd2__ = self.VGAEEncoder2(*args, **kwargs)
        self.__logstd2__ = self.__logstd2__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu2__, self.__logstd2__)
        return z
    
    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu 
  
 
    def kl_loss1(self, mu=None, logstd=None):

        mu = self.__mu1__ if mu is None else mu
        logstd = self.__logstd1__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def kl_loss2(self, mu=None, logstd=None):

        mu = self.__mu2__ if mu is None else mu
        logstd = self.__logstd2__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))        
        
   
    def train_epoch(self, data, test_data, optimizer, alpha, epoch, batch_size=8192, grad_norm=1.0):
#
        x, edge_index = data.x, data.edge_index

        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index, num_nodes=data.num_nodes, num_neg_samples=edge_index.view(2, -1).size(1)
        ).view_as(edge_index)
        N = data.x.shape[0]

        losses = 0
        for perm in DataLoader(range(edge_index.size(1)), batch_size=batch_size, shuffle=True):
            batch_masked_edges = edge_index[:, perm]
            batch_neg_edges = neg_edges[:, perm]
            optimizer.zero_grad()
            z = self.InputEncoder(x, edge_index)
            z1 = self.VGAEEncode1(z, edge_index)
            z2 = self.VGAEEncode2(x, edge_index)
            z0 = torch.cat([z, z1, z2], dim=1)

            pos_out = self.predictor(z0, batch_masked_edges)
            neg_out = self.predictor(z0, batch_neg_edges)
            loss = self.loss_fn(pos_out, neg_out)
            loss = loss + (1.0 / N) * (self.kl_loss1() + self.kl_loss2())

            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()
            losses += loss
        return losses
    
    @torch.no_grad()
    def batch_predict(self, z, edges, batch_size=2 ** 16):
        preds = []
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            preds += [self.predictor(z, edge).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test(self, data, test_data, epoch):

        z = self.get_z(data.x, data.edge_index, self.device)

        pos_edge_index = test_data.pos_edge_label_index
        neg_edge_index = test_data.neg_edge_label_index


        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
        pos_y = torch.ones(pos_pred.size(0))
        neg_y = torch.zeros(neg_pred.size(0))
        y = torch.cat([pos_y, neg_y], dim=0).cpu().numpy()


        auc = roc_auc_score(y, pred)
        aupr = average_precision_score(y, pred)
        binary_pred = (pred >= 0.5).astype(int)  
        acc = accuracy_score(y, binary_pred)
        pre = precision_score(y, binary_pred)
        sen = recall_score(y, binary_pred)  
        F1 = f1_score(y, binary_pred)
        mcc = matthews_corrcoef(y, binary_pred)


        return auc, aupr, acc, pre, sen, F1, mcc, y, pred
    
    @torch.no_grad()
    def get_z1_z2(self, x, edge_index, device):
        z = self.InputEncoder(x, edge_index)
        z1 = self.VGAEEncode1(z, edge_index)
        z2 = self.VGAEEncode2(x, edge_index)
        return z1, z2
    
    @torch.no_grad()
    def get_z(self, x, edge_index, device):
        z = self.InputEncoder(x, edge_index)
        z1 = self.VGAEEncode1(z, edge_index)
        z2 = self.VGAEEncode2(x, edge_index)
        z0 = torch.cat([z, z1, z2], dim=1)
        return z0        