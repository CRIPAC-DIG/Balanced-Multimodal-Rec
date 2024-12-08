import os
import numpy as np
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from utility.parser import parse_args
args = parse_args()


class VBPR(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, image_weight, text_weight, image_feats=None, text_feats=None, adj=None, edge_index=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_image_preference = nn.Embedding(n_users, embedding_dim)
        self.user_text_preference = nn.Embedding(n_users, embedding_dim)


        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.user_image_preference.weight)
        nn.init.xavier_uniform_(self.user_text_preference.weight)

        self.image_feats = torch.Tensor(image_feats).cuda()
        self.text_feats = torch.Tensor(text_feats).cuda()

        self.image_trs = nn.Linear(image_feats.shape[1], embedding_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], embedding_dim)

    def forward(self, training=1):
        id_feats = self.item_embedding.weight
        image_preference = self.user_image_preference.weight
        text_preference = self.user_text_preference.weight
        if training == 1:
            image_feats = self.image_trs(self.image_feats)
            text_feats =  self.text_trs(self.text_feats)
        elif training == 2:
            image_feats = self.image_trs(self.image_feats.mean(dim=0).tile(self.n_items,1))
            image_preference = image_preference.mean(dim=0).tile(self.n_users,1)
            text_feats =  self.text_trs(self.text_feats)
        elif training == 3:
            image_feats = self.image_trs(self.image_feats)
            text_feats =  self.text_trs(self.text_feats.mean(dim=0).tile(self.n_items,1))
            text_preference = text_preference.mean(dim=0).tile(self.n_users,1)

        user_embed = torch.cat([self.user_embedding.weight, image_preference, text_preference], dim=-1)
        item_embed = torch.cat([id_feats, image_feats, text_feats], dim=-1)
        return user_embed, item_embed




class GCN(nn.Module):
    def __init__(self, n_users, adj, embedding_dim):
        super().__init__()
        self.conv_embed_1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_layer1 = nn.Linear(embedding_dim, embedding_dim)
        self.g_layer1 =  nn.Linear(embedding_dim, embedding_dim)    

        self.conv_embed_2 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_layer2 = nn.Linear(embedding_dim, embedding_dim)
        self.g_layer2 =  nn.Linear(embedding_dim, embedding_dim)

        self.conv_embed_3 = nn.Linear(embedding_dim, embedding_dim)
        self.linear_layer3 = nn.Linear(embedding_dim, embedding_dim)
        self.g_layer3 =  nn.Linear(embedding_dim, embedding_dim)
        self.adj = adj

    def forward(self, features, id_embedding, preference):

        x = torch.cat((preference, features),dim=0) 
        x = F.normalize(x).cuda()

        h = F.leaky_relu(torch.mm(self.adj, self.conv_embed_1(x)))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding 
        x = F.leaky_relu(self.g_layer1(h)+x_hat)

        h = F.leaky_relu(torch.mm(self.adj, self.conv_embed_2(x)))
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding 
        x =  F.leaky_relu(self.g_layer2(h)+x_hat)

        h = F.leaky_relu(torch.mm(self.adj, self.conv_embed_3(x)))
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding 
        x =  F.leaky_relu(self.g_layer3(h)+x_hat)

        return x

class MMGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None, adj=None, edge_index=None):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_image_preference = nn.init.xavier_normal_(torch.rand((n_users, embedding_dim), requires_grad=True)).cuda()  #nn.Embedding(n_users, embedding_dim)
        self.user_text_preference = nn.init.xavier_normal_(torch.rand((n_users, embedding_dim), requires_grad=True)).cuda() #nn.Embedding(n_users, embedding_dim)


        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.text_feats = torch.Tensor(text_feats).cuda()
        self.image_feats = torch.Tensor(image_feats).cuda()

        self.image_trs = nn.Linear(image_feats.shape[1], embedding_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], embedding_dim)

        self.v_gcn = GCN(n_users, adj, embedding_dim)
        self.t_gcn = GCN(n_users, adj, embedding_dim)


    def forward(self, training=1):
        id_feats = self.item_embedding.weight
        image_preference = self.user_image_preference#.weight
        text_preference = self.user_text_preference#.weight
        if training == 1:
            image_feats = self.image_trs(self.image_feats)
            text_feats =  self.text_trs(self.text_feats)
        elif training == 2:
            image_feats = self.image_trs(self.image_feats.mean(dim=0).tile(self.n_items,1))
            image_preference = image_preference.mean(dim=0).tile(self.n_users,1)
            text_feats =  self.text_trs(self.text_feats)
        elif training == 3:
            image_feats = self.image_trs(self.image_feats)
            text_feats =  self.text_trs(self.text_feats.mean(dim=0).tile(self.n_items,1))
            text_preference = text_preference.mean(dim=0).tile(self.n_users,1)

        id_embedding = torch.cat([self.user_embedding.weight, id_feats], dim=0)
        t_rep = self.t_gcn(text_feats, id_embedding, text_preference)
        v_rep = self.v_gcn(image_feats, id_embedding, image_preference)
        rep = (t_rep + v_rep) / 2
        user_rep = rep[:self.n_users]
        item_rep = rep[self.n_users:]
        return user_rep, item_rep
