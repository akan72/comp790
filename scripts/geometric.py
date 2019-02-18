import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

# torch_geometric.data.Data describes a single instance of a graph

# One way to specify a Graph
# edge_index = torch.tensor([[0, 1, 1, 2],
#                          [1, 0, 2, 1]], dtype=torch.long)

# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index)

#2nd way to specify a graph (both ways, because graph is undircted)
# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

# data = Data(x=x, edge_index=edge_index.t().contiguous())
# print(data)
# print(data.keys)
# print(data['x'])

# print(data.num_nodes)
# print(data.num_edges)
# print(data.num_features)
# print(data.is_directed())

# Loading benchmark datasets

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import MNISTSuperpixels, TUDataset, ModelNet


dataset  = MNISTSuperpixels(root='../data/MNIST')
split = int(len(dataset) * .9)

# Randomly permute dataset before splitting
dataset = dataset.shuffle()

print(dataset.num_classes)
print(dataset.num_features) 
print(dataset[0])

train = dataset[:split]
test = dataset[split:]

# dataset = ModelNet(root='../data/ModelNet', name='10')
# print(len(dataset))
# print(dataset.num_classes)
# print(dataset.num_features)


# Downloads every time, find way around this? 
# dataset = TUDataset(root='../data/ENZYMES', name='ENZYMES')
# print(len(dataset))
# print(dataset.num_classes)
# print(dataset.num_features)

## TODO: Convert point cloud datasets into graph datasets

# Mini-batches

loader = DataLoader(dataset[:320], batch_size=32, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training= self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train.mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))

# for batch in loader:
#     print(batch)

