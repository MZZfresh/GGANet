import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import getfeature,calculate_metrics,draw_auc, draw_aupr
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
import argparse
from torch.utils.data import DataLoader
from large_model import Transformer
parser = argparse.ArgumentParser()
from torch_geometric.nn import GCNConv, BatchNorm

colors = list(mcolors.TABLEAU_COLORS.keys())



class mlp(torch.nn.Module):
    def __init__(self, num_in, num_hid1, num_hid2, num_out):
        super(mlp, self).__init__()

        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.classify = torch.nn.Linear(num_hid2, num_out)
        self.bn = BatchNorm(num_hid2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.drop(x)
        self.bn(x)
        x = self.classify(x)
        x = self.sigmoid(x)
        return x



class GGANet(nn.Module):
    def __init__(self, data,num_in , num_hid1 , num_hid2, num_out,
                 num_nodes, in_channels, hidden_channels, out_channels, global_dim, 
                 num_layers, heads, ff_dropout, attn_dropout, spatial_size, skip, dist_count_norm,
                   conv_type,num_centroids, no_bn, norm_type ):
        super(GGANet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.large_model = Transformer(num_nodes, in_channels, hidden_channels, out_channels, global_dim, num_layers, heads, ff_dropout, attn_dropout, spatial_size, skip, dist_count_norm, conv_type,num_centroids, no_bn, norm_type)
        self.mlp = mlp(num_in , num_hid1 , num_hid2, num_out )
    def forward(self, data,node_index):
        data.pos_enc = torch.load('poc_enc.pt').to(self.device)
        out_features = self.large_model(data.x, data.edge_index, data.edge_attr, data.pos_enc,batch_idx = node_index)

        return out_features
    def test(self,output,data):
        # 合并特征
        pos_out = []
        neg_out = []
        for d,m in zip(data.pos_edge_label_index[0],data.pos_edge_label_index[1]) :
            drug_feature = output[d]
            mirna_feature = output[m]
            edge_feature = torch.cat((drug_feature, mirna_feature), dim=0)  
            pos_out.append(edge_feature)
        pos_out = torch.stack(pos_out).to(self.device)

        for d,m in zip(data.neg_edge_label_index[0],data.neg_edge_label_index[1]) :
            drug_feature = output[d]
            mirna_feature = output[m]
            edge_feature = torch.cat((drug_feature, mirna_feature), dim=0)  
            neg_out.append(edge_feature)
        neg_out = torch.stack(neg_out).to(self.device)
        # MLP预测部分
        pos_output = self.mlp(pos_out)
        neg_output = self.mlp(neg_out)
        return pos_output , neg_output

    @torch.no_grad()
    def batch_predict(self,z, edges, batch_size=2 ** 16):
        preds = []
        
        for perm in DataLoader(range(edges.size(1)), batch_size):
            edge = edges[:, perm]
            out = []
            for d,m in zip(edge[0],edge[1]) :
                drug_feature = z[d]
                mirna_feature = z[m]
                edge_feature = torch.cat((drug_feature, mirna_feature), dim=0)  
                out.append(edge_feature)
            out = torch.stack(out).to(self.device)

            preds += [self.mlp(out).squeeze().cpu()]

        pred = torch.cat(preds, dim=0)
        return pred

    @torch.no_grad()
    def test_resul(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.batch_predict(z, pos_edge_index)
        neg_pred = self.batch_predict(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y = torch.cat([pos_pred.new_ones(pos_pred.size(0)), neg_pred.new_zeros(neg_pred.size(0))], dim=0)
        y, pred = y.cpu().numpy(), pred.cpu().numpy()

        roc_auc = roc_auc_score(y, pred)
        precision, recall, thresholds = precision_recall_curve(y, pred)
        aupr = auc(recall, precision)
        print('aupr',aupr)
        temp = torch.tensor(pred)
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc ,recall= calculate_metrics(y, temp.cpu())
        return [roc_auc,acc.item(), sen.item(), pre.item(), spe.item(), F1.item(), mcc.item(),recall.item()],y,pred


