from DataHelper import UKBench
import torch
from torch import nn
from torchvision.models import alexnet

# Loss function
def relevance_loss(x, xp, xm):
    loss = 0
    xm = torch.unbind(xm, dim=0)
    for item in xm:
        loss += torch.dist(x, item, 2)
    
    xp = torch.unbind(xp, dim=0)
    for item in xp:
        loss -= torch.dist(x, item, 2)    
    return loss


model = alexnet(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
#model = model.eval()
#model = model.cuda()

dataset = UKBench()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=3)

optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

best_loss = 1000000000
for epoch in range(0, 5):
    for x, xp, xm in dataloader:
        optimizer.zero_grad()

        xf = model(x)
        xpf = model(xp.squeeze(0))
        xmf = model(xm.squeeze(0))
        
        loss = relevance_loss(xf, xpf, xmf)

        loss.backward()
        optimizer.step()
        print(loss.item())
    # Checkpointing
    l = 0
    for i in range(0, 10200, 4):
        x, xp, xm = dataset[i]
        xf = model(x)
        xpf = model(xp.squeeze(0))
        xmf = model(xm.squeeze(0))
        
        l += relevance_loss(xf, xpf, xmf)

    if best_loss > l:
        torch.save(model, "../models/relevance_feedback/model_epoch{}_loss{}.pth".format(epoch, l))    
        best_loss = l