from DataHelper import UKBench
import torch
from torch import nn
from torchvision.models import alexnet

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
model = model.eval()

torch.save(model, "../models/baseline/pretrained_alexnet.pth")
#model = model.cuda()

dataset = UKBench()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=3)

loss = 0
for x, xp, xm in dataloader:
    xf = model(x)
    xpf = model(xp.squeeze(0))
    xmf = model(xm.squeeze(0))
    
    loss += relevance_loss(xf, xpf, xmf)
    print(loss.item())
    
avg_loss = loss.item() / len(dataset)
print(loss)
print(avg_loss)
