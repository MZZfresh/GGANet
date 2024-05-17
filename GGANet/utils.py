import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_data():  
    print("Loading data.")  
    # 加载边信息  
    drug_miRNA = pd.read_csv(r"dataset\edges\edges.csv", header=None)  

    # 特征  
    features = pd.read_csv(r"output.csv", header=None) 
    features = torch.tensor(features.values.tolist())
    # 获取唯一的节点列表  
    miRNA_list = list(set(drug_miRNA[1]))  
    drug_list = list(set(drug_miRNA[0]))  
    # 创建邻接矩阵  
    adj = torch.LongTensor(  
        [[drug_list.index(x[0]), miRNA_list.index(x[1]) + len(drug_list)] for x in drug_miRNA.values]  
    ).T  
    # 创建节点类型张量  
    num_drug_nodes = len(drug_list)  
    num_miRNA_nodes = len(miRNA_list)  
    node_types = torch.cat([torch.zeros(num_drug_nodes, dtype=torch.long), torch.ones(num_miRNA_nodes, dtype=torch.long)]) 
    # 创建节点索引
    node_index = torch.cat([torch.arange(num_drug_nodes), torch.arange(num_miRNA_nodes) + num_drug_nodes])
    # 创建Data对象  
    edge_attr = torch.tensor([1] * adj.size(1), dtype=torch.float)
    data = Data(x=features,node_index = node_index, edge_index=adj, type=node_types,edge_attr=None)  
    # 分割数据集  
    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2, is_undirected=True, split_labels=True, add_negative_train_samples=True)(data)  
    # 将数据集移到GPU（如果可用）  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    train_data = train_data.to(device)  
    test_data = test_data.to(device)  
    # 返回分割后的数据集和邻接矩阵  
    splits = dict(train=train_data, test=test_data, adj=adj, node_types=node_types)  
    return splits

def getfeature(data):  
    # 提取特征和节点类型  
    features = data.x  
    node_types = data['type']  
    # 药物节点类型为0，miRNA节点类型为1  
    drug_node_indices = (node_types == 0).nonzero(as_tuple=True)[0]  
    mirna_node_indices = (node_types == 1).nonzero(as_tuple=True)[0]  
    # 根据节点类型索引提取特征  
    drug_features = features[drug_node_indices]
    mirna_features = features[mirna_node_indices]  
    splits = dict(drug=drug_features, mirna=mirna_features)
    return splits

def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    recall = TP/(TP + FN + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc,recall
def all_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss
def print_result(result):
    metrics = ['auc', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc','recall']
    metric_values = [[] for _ in range(len(metrics))]
    for i in result:
        for j, val in enumerate(i):
            metric_values[j].append(val)
    metric_values = [np.array(m) for m in metric_values]
    formatted_metrics = []
    for metric, values in zip(metrics, metric_values):
        mean = "{:.4f}".format(values.mean())
        std = "{:.4f}".format(np.std(values))
        formatted_metrics.append(f"{metric}: {mean} ± {std}")
    print(*formatted_metrics)
def draw_auc(y, pred, l):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s:AUC = %0.4f' % (l, roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
def draw_aupr(y, pred, l):
    average_precision = average_precision_score(y, pred)
    precision, recall, _ = precision_recall_curve(y, pred)
    plt.plot(recall, precision, label='%s:AUPR = %0.4f' % (l, average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR Curve')
    plt.legend(loc='lower right')
    plt.grid(True)