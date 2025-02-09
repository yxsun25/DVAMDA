import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
import random
import scipy.sparse as sp 
import torch_geometric.transforms as T
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_data_Disbiome():

    dis_sim = np.loadtxt('./dataset/Disbiome/disease_features.txt', delimiter='\t')
    mic_sim = np.loadtxt('./dataset/Disbiome/microbe_features.txt', delimiter='\t')
    adj_triple = np.loadtxt('./dataset/Disbiome/adj.txt')
    dis_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                                    shape=(len(dis_sim), len(mic_sim))).toarray()
    d_emb = torch.FloatTensor(dis_sim)
    m_emb = torch.FloatTensor(mic_sim)
    interaction = pd.DataFrame(dis_mic_matrix)
    return interaction, d_emb, m_emb

def load_data_HMDAD():

    dis_sim = np.loadtxt('./dataset/HMDAD/disease_features.txt', delimiter='\t')
    mic_sim = np.loadtxt('./dataset/HMDAD/microbe_features.txt', delimiter='\t')
    adj_triple = np.loadtxt('./dataset/HMDAD/adj.txt')
    dis_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                                    shape=(len(dis_sim), len(mic_sim))).toarray()
    d_emb = torch.FloatTensor(dis_sim)
    m_emb = torch.FloatTensor(mic_sim)
    interaction = pd.DataFrame(dis_mic_matrix)
    return interaction, d_emb, m_emb


def load_data(database):

    if database == 'Disbiome':
        return load_data_Disbiome()
    elif database == 'HMDAD':
        return load_data_HMDAD()
    else:
        raise ValueError("Invalid database name. Choose from  'Disbiome', 'HMDAD'")

def combine_embeddings(m_emb, s_emb):


    max_length = max(m_emb.size(1), s_emb.size(1))
    m_emb = torch.cat([m_emb, torch.zeros(m_emb.size(0), max_length - m_emb.size(1))], dim=1)
    s_emb = torch.cat([s_emb, torch.zeros(s_emb.size(0), max_length - s_emb.size(1))], dim=1)
    feature = torch.cat([m_emb, s_emb]).cuda()
    return feature


def create_graph_data(interaction, feature, dlen):

    l, p = interaction.values.nonzero()
    adj = torch.LongTensor([p, l + dlen]).cuda()  
    data = Data(x=feature, edge_index=adj).cuda()
    return data


def generate_negative_edges(data, num_neg_samples=None):

    edges = data.edge_index.t().cpu().numpy()
    edge_labels = np.ones((edges.shape[0],))
    num_nodes = data.x.size(0)
    if num_neg_samples is None:
        num_neg_samples = edges.shape[0]
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i!= j and [i, j] not in edges.tolist() and [i, j] not in neg_edges:
            neg_edges.append([i, j])
    neg_edges = np.array(neg_edges)
    return edges, edge_labels, neg_edges

def split_data_into_train_test(data, edges, edge_labels, neg_edges, k_folds=5):
 
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=2025)
    fold_data = []
    for train_idx, test_idx in skf.split(edges, edge_labels):
        train_edges = edges[train_idx]
        test_edges = edges[test_idx]
        train_neg_edges = neg_edges[train_idx]
        test_neg_edges = neg_edges[test_idx]
        train_data = Data(x=data.x, edge_index=torch.tensor(train_edges).t().contiguous().to(torch.long),
                        pos_edge_label_index=torch.tensor(train_edges).t().contiguous().to(torch.long),
                        neg_edge_label_index=torch.tensor(train_neg_edges).t().contiguous().to(torch.long))
        test_data = Data(x=data.x, edge_index=torch.tensor(test_edges).t().contiguous().to(torch.long),
                       pos_edge_label_index=torch.tensor(test_edges).t().contiguous().to(torch.long),
                       neg_edge_label_index=torch.tensor(test_neg_edges).t().contiguous().to(torch.long))
        fold_data.append(dict(train=train_data, test=test_data))
    return fold_data


def prepare_cross_validation_data(database, is_process, k_folds=5):

    #interaction, d_emb, m_emb = load_data_DrugVirus()
    #print(database)
    interaction, d_emb, m_emb = load_data(database)
    feature = combine_embeddings(d_emb, m_emb)
    data = create_graph_data(interaction, feature, len(d_emb))
    edges, edge_labels, neg_edges = generate_negative_edges(data)
    fold_data = split_data_into_train_test(data, edges, edge_labels, neg_edges, k_folds)
    return fold_data, feature.size(1)


if __name__ == "__main__":
    fold_data = prepare_cross_validation_data()