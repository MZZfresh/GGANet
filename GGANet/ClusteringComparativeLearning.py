import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv,GATConv
from info_nce import InfoNCE
from torch_geometric.nn import global_max_pool 
import faiss
import torch.nn.functional as F
import pandas as pd
import numpy as np
drug_miRNA = pd.read_csv(r"dataset\edges\edges.csv", header=None)  
 
miRNA = pd.read_csv(r"dataset\miran\mirna_result_out.csv", header=None)  
drug = pd.read_csv(r"dataset\drug\drug_result_out.csv", header=None)  
 
features = torch.Tensor( drug.values.tolist()+miRNA.values.tolist())  
 
miRNA_list = list(set(drug_miRNA[1]))  
drug_list = list(set(drug_miRNA[0]))  
 
adj = torch.LongTensor(  
    [[drug_list.index(x[0]), miRNA_list.index(x[1]) ] for x in drug_miRNA.values]  
).T  

num_drug_nodes = len(drug_list)  
num_miRNA_nodes = len(miRNA_list)  
node_types = torch.cat([torch.zeros(num_drug_nodes, dtype=torch.long), torch.ones(num_miRNA_nodes, dtype=torch.long)])  




def run_kmeans(x,k):
        """
        Run K-means algorithm to get k clusters of the input tensor x
        """
        latent_dim = x.shape[1]
        kmeans = faiss.Kmeans(d=latent_dim, k=k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze()
        return centroids, node2cluster

def find_positive_pairs(node2cluster):
    positive_pairs = []
    for i in range(len(node2cluster)):
        for j in range(i+1, len(node2cluster)):
            if node2cluster[i] == node2cluster[j]:
                positive_pairs.append((i, j))
    return positive_pairs

def find_negative_pairs(node2cluster):
    negative_pairs = []
    for i in range(len(node2cluster)):
        for j in range(i+1, len(node2cluster)):
            if node2cluster[i] != node2cluster[j]:
                negative_pairs.append((i, j))
    return negative_pairs

def merge_to_edge_index(positive_pairs, negative_pairs):

    pos_edge_index = [[], []]
    neg_edge_index = [[], []]

    for pair in positive_pairs:
        pos_edge_index[0].append(pair[0])  # 起始节点
        pos_edge_index[1].append(pair[1])  # 结束节点


    for pair in negative_pairs:
        neg_edge_index[0].append(pair[0])  # 起始节点
        neg_edge_index[1].append(pair[1])  # 结束节点

    # 将列表转换为PyTorch的LongTensor类型
    pos_edge_index = torch.LongTensor(pos_edge_index)
    neg_edge_index = torch.LongTensor(neg_edge_index)

    return pos_edge_index,neg_edge_index
centroids, node2cluster = run_kmeans(features,10)

positive_pairs = find_positive_pairs(node2cluster)

negative_pairs = find_negative_pairs(node2cluster)
pos_edge_index,neg_edge_index = merge_to_edge_index(positive_pairs, negative_pairs)

data = Data(x=features, edge_index=adj, type=node_types,pos_edge_index =pos_edge_index,neg_edge_index =neg_edge_index) 
graph_data = data
print(graph_data)
train_loader = DataLoader(graph_data, batch_size=64, shuffle=True)

num_features = 192
hidden_dim = 128
num_heads = 1
hidden_features = 128
output_dim = 64
learning_rate = 0.01
num_epochs = 100




class ContrastiveGCN(nn.Module):
    def __init__(self):
        super(ContrastiveGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_features, heads=num_heads)
        self.conv2 = GCNConv(hidden_features, output_dim)
        self.fc2 = nn.Linear(64, 192)
        self.pool = global_max_pool
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        return x


nce_loss = InfoNCE()


model = ContrastiveGCN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(model, optimizer, data, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        anchor_output = model(data.x, data.edge_index)
        positive_output = model(data.x, data.pos_edge_index)
        negative_output = model(data.x, data.neg_edge_index)
        loss = nce_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("Epoch {} - Loss: {:.4f}".format(epoch+1, total_loss))

train(model, optimizer, graph_data, num_epochs)


with torch.no_grad():
    output = model(data.x, data.edge_index)


output_np = output.numpy()


np.savetxt("output.csv", output_np, delimiter=",")



