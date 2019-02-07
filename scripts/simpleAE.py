import torch
import torchvision
from torch import nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle as pkl

from datasets import stackOverflowAdjacencyMatrix
import networkx as nx


num_epochs = 100
batch_size = 128
learning_rate = 1e-3 

dataset = stackOverflowAdjacencyMatrix(path="data/mathOverflow/sx-mathoverflow.txt", nrows=5000)
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
            nn.Sigmoid())
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 


model = autoencoder()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5
    )


outputList = []
for epoch in range(num_epochs):
    for data in dataloader:
        target = data.float()
        target = Variable(target)

        # forward pass
        output = model(target)

        # loss = criterion(output, target.long())
        loss = criterion(output, torch.max(target, 1)[1])

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # training log 


    if epoch % 10 == 0:
        print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))

        outputList.append(output.cpu().data)


pkl.dump([x for x in outputList], open("data/temp/autoencoderOutput.pkl", "wb"))
torch.save(model.state_dict(), './simple_autoencoder.path')





        