import torch
import argparse
from utils import get_data, set_seed,all_loss,print_result
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,roc_curve, auc,precision_recall_curve,average_precision_score,f1_score
import matplotlib.pyplot as plt
from model import GGANet
import random
import torch.nn.functional as F
import os
from tqdm import tqdm
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置可见的CUDA设备，例如使用编号为0的GPU
if torch.cuda.is_available():
    torch.cuda.init()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help="Random seed for model and dataset.")
parser.add_argument('--times', type=int, default=5, help="numbers of training times")
parser.add_argument('--epoch', type=int, default=100, help="numbers of training epoch")
parser.add_argument('--lr', type=float, default=0.001, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=0.001, help='weight decay in optimizer')
# mlp
parser.add_argument('--num_in', type=int, default=64, help="mlp num_in")
parser.add_argument('--num_hid1', type=int, default=32, help="mlp num_hid1")
parser.add_argument('--num_hid2', type=int, default=16, help="mlp num_hid2")  
parser.add_argument('--num_out', type=int, default=1, help="mlp num_out")
# larg_model
 # training
parser.add_argument('--sizes', type=int, nargs='+', default=[30,15,5,1]) 
parser.add_argument('--test_sizes', type=int, nargs='+', default=[30,15,5,1]) 
 # NN 
parser.add_argument('--conv_type', type=str, default='full', choices=['local', 'global', 'full'])
parser.add_argument('--in_channels', type=int, default=192) 
parser.add_argument('--out_channels', type=int, default=32) 
parser.add_argument('--hidden_dim', type=int, default=64) 
parser.add_argument('--global_dim', type=int, default=128) 
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=4) 
parser.add_argument('--attn_dropout', type=float, default=0.2)
parser.add_argument('--ff_dropout', type=float, default=0.2)
parser.add_argument('--skip', type=int, default=10)  
parser.add_argument('--dist_count_norm', type=int, default=1)  
parser.add_argument('--num_centroids', type=int, default=5) 
parser.add_argument('--no_bn', action='store_true')
parser.add_argument('--norm_type', type=str, default='batch_norm',choices=['layer_norm','batch_norm'])
args = parser.parse_args()
print(args)
set_seed(args.seed)

# 聚类对比学习
subprocess.run(["python", "ClusteringComparativeLearning.py"])



data = get_data()

model = GGANet(data=data['train'] ,
        num_nodes=data['train'].num_nodes,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_dim, 
        out_channels=args.out_channels,
        global_dim=args.global_dim,
        num_layers=args.num_layers,
        heads=args.num_heads,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        spatial_size=len(args.sizes),
        skip=args.skip,
        dist_count_norm=args.dist_count_norm,
        conv_type=args.conv_type,
        num_centroids=args.num_centroids,
        no_bn=args.no_bn,
        norm_type=args.norm_type,
        num_in=args.num_in,
        num_hid1=args.num_hid1,
        num_hid2=args.num_hid2,
        num_out=args.num_out) 

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
model.to('cuda:0')
data = {key: value.to('cuda:0') for key, value in data.items()}



all_result = []
for i in tqdm(range(args.times), desc="Total Progress",leave=False):
    data_train = data['train']
    for epoch in tqdm(range(args.epoch), desc=f"Epoch Progress (Iteration {i+1}/{args.times})", leave=False):
        model.train()
        optimizer.zero_grad()
        data_in = data_train
        output = model(data_in,data_in.node_index)
        pos_output , neg_output = model.test(output,data_in)
        loss = all_loss (pos_output, neg_output)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{args.epoch}, Loss: {loss.item()}")

    model.eval()
    data2 = data['train]
    test_data = data['test']
    data_in = test_data
    z = model(data2,data2.node_index)
    a,b = model.test(z,data_in)
    result ,y, pred = model.test_resul(z, data_in.pos_edge_label_index, data_in.neg_edge_label_index)

    fpr, tpr, _ = roc_curve(y, pred)

    l = (i+1)*args.epoch
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    plt.plot(fpr, tpr, label='%s:AUC = %0.4f' % (l, roc_auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.figure(2)
    average_precision = average_precision_score(y, pred)
    precision, recall, _ = precision_recall_curve(y, pred)
    plt.plot(recall, precision, label='%s:AUPR = %0.4f' % (l, average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR Curve')
    plt.legend(loc='lower right')
    plt.grid(True)



    all_result.append(result)
    print_result(all_result)
plt.show()
