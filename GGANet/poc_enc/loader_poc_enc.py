import argparse
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

from utils import get_data

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset_name = 'ogbn-arxiv'
    _, dataset = get_data()  # 注意这里使用了 _ 占位符来忽略第一个返回值
    print(dataset)
    data = dataset.to(device)  # 将数据移动到正确的设备上

    # 定义 Node2Vec 模型
    model = Node2Vec(data.edge_index, embedding_dim=args.embedding_dim,
                     walk_length=args.walk_length, context_size=args.context_size,
                     walks_per_node=args.walks_per_node, num_negative_samples=1)

    # 训练模型
    model.train()

    # 获取节点嵌入
    embeddings = model.embedding.weight

    # 将节点嵌入保存到文件中
    torch.save(embeddings, 'node_embeddings.pt')

if __name__ == "__main__":
    main()
    import torch

# 加载 .pt 文件
embeddings = torch.load('node_embeddings.pt')

# 查看张量的内容
print(embeddings.shape)

