import torch
import torchvision
from torch import nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import stackOverflowAdjacencyMatrix
import networkx as nx


num_epochs = 300
batch_size = 128
learning_rate = 1e-3 

dataset = stackOverflowAdjacencyMatrix(path="data/mathOverflow/sx-mathoverflow.txt", nrows=450)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

size = len(dataset.adjacency)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, size),
            nn.Tanh())
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 



model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

for epoch in range(num_epochs):
    for data in dataloader:
        row = data.float()
        row = Variable(row)

        # forward pass
        output = model(row)
        loss = criterion(output, row)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # training log 
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))

torch.save(model.state_dict(), './simple_autoencoder.path')



        